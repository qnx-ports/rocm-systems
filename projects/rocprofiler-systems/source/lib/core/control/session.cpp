// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::control
{
session::session() noexcept
{
    for(auto& a : m_active)
        a.store(true, std::memory_order_relaxed);
}

void
session::shutdown()
{
    {
        std::scoped_lock const lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }
    {
        std::scoped_lock const lk{ m_actions_mutex };
        m_actions.clear();
        for(auto& a : m_active)
            a.store(true, std::memory_order_relaxed);
    }
}

void
session::subscribe(subscriber sub)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    m_subscribers.push_back(std::move(sub));
}

void
session::register_trigger(std::string_view name, action initial, scope event_scope)
{
    std::scoped_lock const lk{ m_actions_mutex };
    m_actions[std::string{ name }] = entry{ initial, event_scope };
    update_active_locked(event_scope);
}

void
session::unregister_trigger(std::string_view name)
{
    std::scoped_lock const lk{ m_actions_mutex };

    const auto it = m_actions.find(std::string{ name });
    if(it == m_actions.end()) return;

    const auto event_scope = it->second.event_scope;
    m_actions.erase(it);
    update_active_locked(event_scope);
}

void
session::set_action(std::string_view name, action act)
{
    // Serializes compute-then-notify across concurrent callers so subscribers
    // observe transitions in the same order they were computed. Deliberately
    // a separate mutex from m_actions_mutex (released below, before
    // notify_pause()/notify_resume() run) so a subscriber callback that
    // re-enters is_active()/is_active_without() cannot deadlock.
    std::scoped_lock const notify_lk{ m_notify_mutex };

    scope event_scope = scope::global;
    bool  was_active  = false;
    bool  now_active  = false;
    {
        std::scoped_lock const lk{ m_actions_mutex };

        const auto it = m_actions.find(std::string{ name });
        if(it == m_actions.end()) return;

        event_scope          = it->second.event_scope;
        const auto scope_idx = static_cast<std::size_t>(event_scope);

        was_active     = m_active[scope_idx].load(std::memory_order_relaxed);
        it->second.act = act;
        now_active     = resolve_locked(event_scope);
        m_active[scope_idx].store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active) return;

    LOG_DEBUG("session: trigger '{}' {} its scope", name,
              now_active ? "resumed" : "paused");

    if(now_active)
        notify_resume(event_scope);
    else
        notify_pause(event_scope);
}

void
session::force_initial_pause()
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        const bool any_paused_for_sub =
            std::any_of(sub.scopes.begin(), sub.scopes.end(),
                        [this](scope listened) { return !is_active(listened); });

        if(any_paused_for_sub && sub.on_pause) sub.on_pause();
    }
}

void
session::update_active_locked(scope event_scope)
{
    const auto idx = static_cast<std::size_t>(event_scope);
    assert(idx < scope_count);
    m_active[idx].store(resolve_locked(event_scope), std::memory_order_relaxed);
}

// Any pause action within the given scope pauses that scope. Skip is
// ignored. With no actions for the scope, the scope is active by default.
bool
session::resolve_locked(scope event_scope) const noexcept
{
    return std::none_of(
        m_actions.begin(), m_actions.end(), [event_scope](const auto& kv) {
            return kv.second.event_scope == event_scope && kv.second.act == action::pause;
        });
}

bool
session::is_active_without(std::string_view name, scope event_scope) const noexcept
{
    std::scoped_lock const lk{ m_actions_mutex };
    return std::none_of(m_actions.begin(), m_actions.end(),
                        [name, event_scope](const auto& kv) {
                            return kv.second.event_scope == event_scope &&
                                   kv.first != name && kv.second.act == action::pause;
                        });
}

namespace
{
bool
listens_to(const subscriber& sub, scope event_scope)
{
    return std::find(sub.scopes.begin(), sub.scopes.end(), event_scope) !=
           sub.scopes.end();
}
}  // namespace

void
session::notify_pause(scope event_scope)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(!listens_to(sub, event_scope)) continue;
        LOG_DEBUG("session: pausing subscriber '{}'", sub.name);
        if(sub.on_pause) sub.on_pause();
    }
}

void
session::notify_resume(scope event_scope)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(!listens_to(sub, event_scope)) continue;
        const bool all_active =
            std::all_of(sub.scopes.begin(), sub.scopes.end(),
                        [this](scope listened) { return is_active(listened); });
        if(!all_active) continue;
        LOG_DEBUG("session: resuming subscriber '{}'", sub.name);
        if(sub.on_resume) sub.on_resume();
    }
}
}  // namespace rocprofsys::control

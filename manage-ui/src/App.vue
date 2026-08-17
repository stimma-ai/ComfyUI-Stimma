<template>
  <div class="app" style="position:relative" ref="rootEl">
    <div class="head">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="color:var(--text2)" aria-hidden="true"><path d="M5.485 23.76c-.568 0-1.026-.207-1.325-.598-.307-.402-.387-.964-.22-1.54l.672-2.315a.605.605 0 00-.1-.536.622.622 0 00-.494-.243H2.085c-.568 0-1.026-.207-1.325-.598-.307-.403-.387-.964-.22-1.54l2.31-7.917.255-.87c.343-1.18 1.592-2.14 2.786-2.14h2.313c.276 0 .519-.18.595-.442l.764-2.633C9.906 1.208 11.155.249 12.35.249l4.945-.008h3.62c.568 0 1.027.206 1.325.597.307.402.387.964.22 1.54l-1.035 3.566c-.343 1.178-1.593 2.137-2.787 2.137l-4.956.01H11.37a.618.618 0 00-.594.441l-1.928 6.604a.605.605 0 00.1.537c.118.153.3.243.495.243l3.275-.006h3.61c.568 0 1.026.206 1.325.598.307.402.387.964.22 1.54l-1.036 3.565c-.342 1.179-1.592 2.138-2.786 2.138l-4.957.01h-3.61z"/></svg>
      <span class="name">ComfyUI</span>
      <span class="sp"></span>
      <span v-if="overview?.state !== 'in_progress'" class="state"><span class="dot" :class="stateDot"></span>{{ stateLabel }}</span>
    </div>
    <div class="tabs">
      <button
        v-for="t in tabs"
        :key="t.id"
        :class="{
          on: tab === t.id,
          attention: t.id === 'activity' && overview?.activity_attention_count,
        }"
        @click="tab = t.id"
      >
        {{ t.label }}
        <template v-if="t.id === 'activity'">
          <span v-if="overview?.activity_attention_count" class="dot a" title="Activity needs attention"></span>
          <span v-else-if="overview?.activity_active_count" class="spin tab-spin" title="Activity in progress"></span>
        </template>
      </button>
    </div>
    <div v-if="updateAvailable || updateBusy" class="update-notice" :class="{ running: updateBusy }">
      <span v-if="updateBusy" class="spin update-spin" aria-hidden="true"></span>
      <svg v-else viewBox="0 0 24 24" aria-hidden="true"><path d="M12 18V6m0 0-4.5 4.5M12 6l4.5 4.5" /></svg>
      <div class="update-text">
        <div>{{ updateBusy ? 'Updating ComfyUI-Stimma' : 'ComfyUI-Stimma update available' }}</div>
        <div class="mono">{{ updateHashes }}</div>
      </div>
      <button v-if="!updateBusy" class="btn sm update-cta" @click="applyUpdate">Update & restart</button>
    </div>
    <div v-if="overview && overview.summary && !['ready', 'in_progress'].includes(overview.state)" class="banner" :class="{ red: overview.state === 'error' }">
      <span>{{ bannerText }}</span>
      <button v-if="overview.restart_needed && overview.restart_needed.length" class="lnk" :disabled="restartBusy" @click="restart">{{ restartBusy ? 'Restarting…' : 'Restart' }}</button>
      <button v-else-if="overview.state !== 'ready'" class="lnk" @click="tab = overview.summary.toLowerCase().includes('download') ? 'activity' : 'overview'">Details</button>
    </div>
    <div v-else-if="overview && overview.restart_needed && overview.restart_needed.length" class="banner">
      <span>Restart ComfyUI to finish setup</span>
      <button class="lnk" :disabled="restartBusy" @click="restart">{{ restartBusy ? 'Restarting…' : 'Restart' }}</button>
    </div>
    <div v-if="restartError" class="banner red"><span>{{ restartError }}</span><button class="lnk" @click="restartError = ''">Dismiss</button></div>
    <div v-if="managerNotice" class="capability" :class="{ red: managerNotice.state === 'failed' }">
      <div class="cap-text">
        <div>{{ managerNotice.title }}</div>
        <div v-if="managerNotice.detail" class="cap-detail">{{ managerNotice.detail }}</div>
      </div>
      <button v-if="managerNotice.state === 'missing'" class="btn sm primary" :disabled="managerBusy" @click="installManager">{{ managerBusy ? 'Starting…' : 'Install' }}</button>
      <button v-else-if="managerNotice.state === 'failed'" class="btn sm" :disabled="managerBusy" @click="installManager">{{ managerBusy ? 'Retrying…' : 'Retry' }}</button>
      <button v-else class="lnk" @click="tab = 'activity'">Activity</button>
    </div>

    <OverviewTab v-if="tab === 'overview'" :overview="overview" @refresh="load" @open="tab = $event" />
    <WorkflowsTab v-else-if="tab === 'workflows'" @activity="tab = 'activity'" />
    <ActivityTab v-else-if="tab === 'activity'" />
    <SettingsTab v-else :overview="overview" @refresh="load" />
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, nextTick, watch } from 'vue'
import { api } from './api'
import OverviewTab from './OverviewTab.vue'
import WorkflowsTab from './WorkflowsTab.vue'
import ActivityTab from './ActivityTab.vue'
import SettingsTab from './SettingsTab.vue'

const tabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'workflows', label: 'Workflows' },
  { id: 'activity', label: 'Activity' },
  { id: 'settings', label: 'Settings' },
]
const initial = (location.hash || '').replace('#', '')
const tab = ref(tabs.some(t => t.id === initial) ? initial : 'overview')
const overview = ref(null)
const managerBusy = ref(false)
const updateBusy = ref(false)
const restartBusy = ref(false)
const restartError = ref('')
let timer = null
let hostRefreshTimer = null

async function load() {
  try { overview.value = await api.overview() } catch (e) { overview.value = overview.value || { state: 'error', summary: 'Manager unavailable' } }
}
const rootEl = ref(null)
let sizeObs = null
let lastSent = 0
// Tell an embedding host (the Stimma popover) how tall we'd like to be so it
// can size the iframe to content instead of a fixed height.
function reportSize() {
  const root = rootEl.value
  if (!root || window.parent === window) return
  const body = root.querySelector('.body')
  let h = 0
  for (const el of root.children) {
    if (el === body) {
      // Content height, not the flex-allotted box (scrollHeight >= clientHeight)
      let inner = 12 // .body vertical padding
      for (const c of body.children) inner += c.offsetHeight || 0
      h += inner
    }
    else if (!el.classList.contains('sheet-wrap')) h += el.offsetHeight || 0
  }
  const sheet = root.querySelector('.sheet')
  if (sheet) h = Math.max(h, sheet.offsetHeight + 80)
  h = Math.ceil(h)
  if (Math.abs(h - lastSent) < 2) return
  lastSent = h
  try { window.parent.postMessage({ type: 'stimma-manage-size', height: h }, '*') } catch { /* */ }
}
onMounted(() => {
  load(); timer = setInterval(load, 3000)
  sizeObs = new MutationObserver(() => nextTick(reportSize))
  if (rootEl.value) sizeObs.observe(rootEl.value, { childList: true, subtree: true, characterData: true, attributes: true })
  nextTick(reportSize)
  window.addEventListener('resize', reportSize)
  window.addEventListener('message', onHostMessage)
})
onUnmounted(() => {
  clearInterval(timer)
  if (hostRefreshTimer) clearTimeout(hostRefreshTimer)
  sizeObs?.disconnect()
  window.removeEventListener('resize', reportSize)
  window.removeEventListener('message', onHostMessage)
})
watch(tab, () => nextTick(reportSize))
window.addEventListener('hashchange', () => { const h = location.hash.replace('#', ''); if (tabs.some(t => t.id === h)) tab.value = h })

function onHostMessage(e) {
  if (e.data?.type !== 'stimma-manage-refresh') return
  // Progress can arrive around 10 times a second. Coalesce it into a prompt
  // manager refresh without reintroducing a multi-second polling delay.
  if (hostRefreshTimer) return
  hostRefreshTimer = setTimeout(() => {
    hostRefreshTimer = null
    load()
  }, 100)
}

const stateDot = computed(() => overview.value?.checking ? 'z' : ({ ready: 'g', warning: 'a', error: 'r' }[overview.value?.state] || 'z'))
const updateAvailable = computed(() => !!overview.value?.plugin?.update_available)
const updateHashes = computed(() => {
  const plugin = overview.value?.plugin
  if (!plugin) return ''
  if (updateBusy.value) return plugin.head || ''
  return [plugin.head, plugin.target].filter(Boolean).join(' → ')
})
const stateLabel = computed(() => {
  if (overview.value?.checking) return 'Checking'
  if (overview.value?.plugin?.restart_required) return 'Restart required'
  const s = overview.value?.state
  if (!s) return '…'
  if (s === 'ready') return 'Ready'
  if (s === 'warning') return 'Degraded'
  return 'Error'
})
const bannerText = computed(() => overview.value?.summary || '')
const managerNotice = computed(() => {
  const m = overview.value?.comfyui_manager
  if (!m || m.state === 'ready' || m.state === 'restart_needed') return null
  if (m.state === 'installing') return { state: m.state, title: 'Installing ComfyUI-Manager', detail: m.operation?.detail || '' }
  if (m.state === 'failed') return { state: m.state, title: 'ComfyUI-Manager install failed', detail: m.operation?.error || '' }
  return { state: 'missing', title: 'Install ComfyUI-Manager to enable automatic custom-node setup.', detail: '' }
})
async function installManager() {
  managerBusy.value = true
  try { await api.installManager(); await load() }
  catch (e) { overview.value = { ...(overview.value || {}), comfyui_manager: { state: 'failed', operation: { error: e.message } } } }
  finally { managerBusy.value = false }
}
async function applyUpdate() {
  updateBusy.value = true
  try {
    const result = await api.updateApply()
    if (!result.ok) throw new Error(result.error || 'Update failed')
    if (!result.restarting) await load()
  } catch (e) {
    alert(e.message)
  } finally {
    updateBusy.value = false
  }
}
async function restart() {
  restartBusy.value = true
  restartError.value = ''
  try { await api.restart('all') }
  catch (e) {
    restartError.value = e.message
    restartBusy.value = false
  }
}
</script>

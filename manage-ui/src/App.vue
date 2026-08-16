<template>
  <div class="app" style="position:relative" ref="rootEl">
    <div class="head">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="color:var(--text2)" aria-hidden="true"><path d="M5.485 23.76c-.568 0-1.026-.207-1.325-.598-.307-.402-.387-.964-.22-1.54l.672-2.315a.605.605 0 00-.1-.536.622.622 0 00-.494-.243H2.085c-.568 0-1.026-.207-1.325-.598-.307-.403-.387-.964-.22-1.54l2.31-7.917.255-.87c.343-1.18 1.592-2.14 2.786-2.14h2.313c.276 0 .519-.18.595-.442l.764-2.633C9.906 1.208 11.155.249 12.35.249l4.945-.008h3.62c.568 0 1.027.206 1.325.597.307.402.387.964.22 1.54l-1.035 3.566c-.343 1.178-1.593 2.137-2.787 2.137l-4.956.01H11.37a.618.618 0 00-.594.441l-1.928 6.604a.605.605 0 00.1.537c.118.153.3.243.495.243l3.275-.006h3.61c.568 0 1.026.206 1.325.598.307.402.387.964.22 1.54l-1.036 3.565c-.342 1.179-1.592 2.138-2.786 2.138l-4.957.01h-3.61z"/></svg>
      <span class="name">ComfyUI</span>
      <span class="sp"></span>
      <span class="state"><span class="dot" :class="stateDot"></span>{{ stateLabel }}</span>
    </div>
    <div class="tabs">
      <button v-for="t in tabs" :key="t.id" :class="{ on: tab === t.id }" @click="tab = t.id">{{ t.label }}</button>
    </div>
    <div v-if="overview && overview.summary && overview.state !== 'ready'" class="banner" :class="{ red: overview.state === 'error' }">
      <span>{{ bannerText }}</span>
      <button v-if="overview.restart_needed && overview.restart_needed.length" class="lnk" @click="restart">Restart</button>
      <button v-else-if="overview.state !== 'ready'" class="lnk" @click="tab = overview.summary.toLowerCase().includes('download') ? 'activity' : 'overview'">Details</button>
    </div>
    <div v-else-if="overview && overview.restart_needed && overview.restart_needed.length" class="banner">
      <span>Restart ComfyUI to finish setup.</span>
      <button class="lnk" @click="restart">Restart</button>
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
let timer = null

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
})
onUnmounted(() => { clearInterval(timer); sizeObs?.disconnect(); window.removeEventListener('resize', reportSize) })
watch(tab, () => nextTick(reportSize))
window.addEventListener('hashchange', () => { const h = location.hash.replace('#', ''); if (tabs.some(t => t.id === h)) tab.value = h })

const stateDot = computed(() => ({ ready: 'g', warning: 'a', error: 'r' }[overview.value?.state] || 'z'))
const stateLabel = computed(() => {
  const s = overview.value?.state
  if (!s) return '…'
  if (s === 'ready') return 'Ready'
  if (s === 'warning') return 'Degraded'
  return 'Error'
})
const bannerText = computed(() => overview.value?.summary || '')
async function restart() {
  if (!confirm('Restart ComfyUI now? Running jobs will be interrupted.')) return
  try { await api.restart('all') } catch (e) { alert(e.message) }
}
</script>

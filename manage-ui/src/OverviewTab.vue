<template>
  <div class="body">
    <template v-if="!overview"><div class="empty"><span class="spin"></span></div></template>
    <template v-else>
      <div v-for="h in overview.hosts" :key="h.host" class="grp">
        <h4>{{ hostLabel(h) }} <span class="n">{{ hostSub(h) }}</span></h4>
        <div v-if="h.gpus && h.gpus.length" class="gpus">
          <div v-for="g in h.gpus" :key="g.uuid || g.index" class="gpu" :class="{ down: !h.reachable }">
            <div class="k" :title="g.name">GPU {{ g.index }}</div>
            <div class="v mono">{{ h.reachable && g.util != null ? Math.round(g.util) + '%' : '—' }}</div>
            <div class="bar"><i :style="{ width: memPct(g) + '%' }"></i></div>
            <div class="k mono" style="margin-top:3px">{{ h.reachable ? mem(g) : '—' }}</div>
          </div>
        </div>
        <div v-else class="li"><div class="t"><div class="b">{{ h.checking ? 'Checking…' : h.reachable ? '' : 'Unreachable' }}</div></div></div>
      </div>

      <div class="grp">
        <h4>Running <span class="n" v-if="!overview.running.length">nothing right now</span></h4>
        <div v-for="j in overview.running" :key="j.prompt_id" class="li">
          <div class="t">
            <div class="a">{{ j.title }}</div>
            <div class="bar" :class="{ ind: j.progress == null }"><i :style="{ width: (j.progress != null ? Math.round(j.progress * 100) : 40) + '%' }"></i></div>
            <div class="b" style="margin-top:4px">{{ jobSub(j) }}</div>
          </div>
          <div class="r"><button class="btn sm ghost" title="Cancel" @click="cancel(j)">✕</button></div>
        </div>
      </div>
      <div class="grp" v-if="pendingTotal">
        <h4>Queued <span class="n mono">{{ pendingTotal }}</span></h4>
      </div>
    </template>
  </div>
  <div class="foot" v-if="overview">
    <span>ComfyUI-Stimma <span class="mono">{{ pluginLabel }}</span></span>
    <span class="sp"></span>
    <button v-if="overview.plugin && overview.plugin.update_available" @click="$emit('open', 'settings')" style="color:var(--accent-hi)">Update</button>
    <span v-else-if="overview.disk && overview.disk.length" class="mono">{{ fmtBytes(overview.disk[0].free) }} free</span>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { api, fmtBytes, fmtElapsed } from './api'
const props = defineProps({ overview: Object })
const emit = defineEmits(['refresh', 'open'])

const pendingTotal = computed(() => (props.overview?.pending || 0) + (props.overview?.stp_queue?.queued || 0))
const pluginLabel = computed(() => {
  const p = props.overview?.plugin
  if (!p) return ''
  if (!p.git) return p.version
  if (p.update_available) return `${p.head} · ${p.behind} behind`
  return p.head || ''
})
function hostLabel(h) { return h.local ? (h.hostname || 'This machine') : h.host }
function hostSub(h) {
  const n = h.instances.length
  const up = h.instances.filter(i => i.healthy).length
  if (h.checking && !h.reachable) return 'checking'
  if (!h.reachable) { const ls = h.instances[0]?.last_seen; return 'unreachable' + (ls ? ` · last seen ${ago(ls)}` : '') }
  return `${n} instance${n === 1 ? '' : 's'}${up < n ? ` · ${n - up} down` : ''}`
}
function ago(ts) { const s = Date.now() / 1000 - ts; return s < 90 ? 'just now' : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${Math.floor(s / 3600)} h ago` }
function memPct(g) { return g.mem_total ? Math.min(100, Math.round(100 * (g.mem_used || 0) / g.mem_total)) : 0 }
function mem(g) { return g.mem_total ? `${Math.round((g.mem_used || 0) / 2 ** 30)} / ${Math.round(g.mem_total / 2 ** 30)} GB` : '' }
function jobSub(j) {
  const parts = []
  if (j.started_at) parts.push(fmtElapsed(j.started_at) + ' elapsed')
  if (!j.ours) parts.push('ComfyUI')
  if (props.overview.hosts.length > 1 || (props.overview.hosts[0]?.instances.length > 1)) parts.push(j.addr)
  return parts.join(' · ')
}
async function cancel(j) { try { await api.cancelJob(j.prompt_id, j.addr); emit('refresh') } catch (e) { alert(e.message) } }
</script>

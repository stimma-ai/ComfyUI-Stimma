<template>
  <template v-if="detail">
    <div class="subhead">
      <button class="back" @click="detail = null">‹</button>
      <span>{{ detail.name || detail.slug }}</span>
      <span class="sp" style="flex:1"></span>
      <span class="dot" :class="detail.state === 'ready' ? 'g' : 'z'" style="margin-right:2px"></span>
    </div>
    <div class="body">
      <div v-if="detail.loading" class="empty"><span class="spin"></span></div>
      <template v-else>
        <div class="grp" v-if="detail.models && detail.models.length">
          <h4>Models</h4>
          <div v-for="m in detail.models" :key="m.filename" class="li" style="min-height:32px;padding:6px 0">
            <span class="dot" :class="m.installed ? 'g' : 'z'"></span>
            <div class="t">
              <div class="a mono" :title="m.filename">{{ m.filename }}</div>
              <div class="b">{{ modelSub(m) }}</div>
            </div>
            <div class="r mono">{{ m.size ? fmtBytes(m.size) : '' }}</div>
          </div>
        </div>
        <div class="grp" v-if="detail.packs && detail.packs.length">
          <h4>Node packs</h4>
          <div v-for="pk in detail.packs" :key="pk.class_type" class="li" style="min-height:32px;padding:6px 0">
            <span class="dot z"></span>
            <div class="t"><div class="a">{{ pk.title || pk.class_type }}</div><div class="b mono">{{ pk.class_type }}</div></div>
            <div class="r">missing</div>
          </div>
        </div>
        <div class="grp" v-if="!detail.error && !(detail.models || []).length && !(detail.packs || []).length"><div class="empty">No dependencies</div></div>
        <div v-if="detail.error" class="empty">{{ detail.error }}</div>
      </template>
    </div>
    <div class="foot" v-if="detail.state === 'needs_setup'">
      <span class="mono">{{ detail.file }}</span>
      <span class="sp"></span>
      <button v-if="detail.in_progress" style="color:var(--accent-hi)" @click="detail = null; $emit('activity')">Activity</button>
      <button v-else class="btn sm" @click="openPlanFromDetail">Get ready</button>
    </div>
    <div class="foot" v-else><span class="mono">{{ detail.file }}</span></div>
  </template>
  <template v-else>
  <div class="pills">
    <button v-for="f in filters" :key="f.id" class="pill" :class="{ on: filter === f.id }" @click="filter = f.id">{{ f.label }}<span class="n">{{ f.count }}</span></button>
  </div>
  <div class="body">
    <div v-if="!data" class="empty"><span class="spin"></span></div>
    <template v-else>
      <div class="grp" v-if="rows.length">
        <div v-for="w in rows" :key="w.key" class="li" :class="{ dim: w.kind === 'other' }" :style="w.kind === 'tool' ? 'cursor:pointer' : ''" @click="w.kind === 'tool' && openDetail(w)">
          <span class="dot" :class="dotFor(w)"></span>
          <div class="t">
            <div class="a">{{ w.name }}</div>
            <div class="b">{{ subFor(w) }}</div>
          </div>
          <div class="r">
            <button v-if="w.state === 'needs_setup' && !w.in_progress" class="btn sm" @click.stop="openPlan(w)">Get ready</button>
            <button v-else-if="w.in_progress" class="btn sm ghost" style="color:var(--accent-hi)" @click.stop="$emit('activity')">Activity</button>
            <span v-else-if="w.kind === 'tool'" style="color:var(--muted)">›</span>
          </div>
        </div>
      </div>
      <div v-else class="empty">None</div>
    </template>
  </div>
  <div class="foot">
    <span class="sp"></span>
    <button :disabled="scanning" @click="rescan">{{ scanning ? 'Scanning…' : 'Rescan' }}</button>
  </div>
  </template>

  <div v-if="sheet" class="sheet-wrap" @click.self="closeSheet">
    <div class="sheet">
      <template v-if="sheet.loading">
        <h5>{{ sheet.w.name }}</h5>
        <p><span class="spin"></span></p>
      </template>
      <template v-else-if="sheet.error">
        <h5>Setup unavailable</h5>
        <p>{{ sheet.error }}</p>
        <div class="acts"><button class="btn" @click="closeSheet">Close</button></div>
      </template>
      <template v-else>
        <h5>Set up {{ sheet.plan.name }}</h5>
        <p v-if="downloadsToDo.length">Downloads to {{ targetsLabel }} · {{ fmtBytes(sheet.plan.total_size) }}<span v-if="sheet.plan.free_space"> · {{ fmtBytes(sheet.plan.free_space) }} free</span></p>
        
        <div class="lst" v-if="downloadsToDo.length">
          <div v-for="d in downloadsToDo" :key="d.filename"><span class="mono f" :title="d.filename">{{ d.filename }}</span><span class="mono">{{ d.size ? fmtBytes(d.size) : (d.resolved ? '' : 'no source') }}</span></div>
        </div>
        <div class="lst" v-if="packsToDo.length">
          <div v-for="p in packsToDo" :key="p.url"><span class="f">Install {{ p.title }}</span><span class="mono">{{ p.installable ? 'node pack' : 'manual' }}</span></div>
        </div>

        <!-- blockers -->
        <div v-for="(b, i) in sheet.plan.blockers" :key="i" class="warn">
          <template v-if="b.kind === 'hf_token'">
            Needs a <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Hugging Face read token ↗</a><template v-if="b.license_url"> and the <a :href="b.license_url" target="_blank" rel="noopener">model license ↗</a> accepted</template>.
            <div class="field"><input v-model="hfToken" type="password" placeholder="hf_…" spellcheck="false" /></div>
          </template>
          <template v-else-if="b.kind === 'hf_license'">
            {{ b.repo || 'This model' }} is gated — <a :href="b.license_url" target="_blank" rel="noopener">accept the license on Hugging Face ↗</a>.
          </template>
          <template v-else-if="b.kind === 'no_source'">
            No known source for <span class="mono">{{ b.filename }}</span>.
            <div class="field"><input v-model="sources[b.filename]" placeholder="Download URL" spellcheck="false" /></div>
          </template>
          <template v-else-if="b.kind === 'unknown_node'">
            <span class="mono">{{ b.class_type }}</span> — no known node pack.
          </template>
          <template v-else-if="b.kind === 'no_manager'">
            ComfyUI-Manager not installed — run on the ComfyUI machine, then restart:
            <div v-for="p in b.packs" :key="p.url" class="cmd mono">{{ p.manual }}</div>
          </template>
        </div>

        <div class="acts">
          <button class="btn" @click="closeSheet">Cancel</button>
          <button class="btn primary" :disabled="!canStart || starting" @click="start">{{ starting ? 'Starting…' : startLabel }}</button>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { api, fmtBytes } from './api'
const emit = defineEmits(['activity'])

const data = ref(null)
const filter = ref('all')
const scanning = ref(false)
const sheet = ref(null)
const detail = ref(null)
const hfToken = ref('')
const sources = reactive({})
const starting = ref(false)
let timer = null

async function load() { try { data.value = await api.workflows() } catch (e) { /* keep last */ } }
onMounted(() => { load(); timer = setInterval(load, 4000) })
onUnmounted(() => clearInterval(timer))

const all = computed(() => {
  if (!data.value) return []
  const tools = data.value.tools.map(t => ({ ...t, kind: 'tool', key: 't:' + t.slug }))
  const others = data.value.others.map(o => ({ ...o, kind: 'other', name: o.file.replace(/\.json$/, ''), key: 'o:' + o.path }))
  return [...tools, ...others]
})
const filters = computed(() => [
  { id: 'all', label: 'All', count: all.value.length },
  { id: 'ready', label: 'Ready', count: all.value.filter(w => w.state === 'ready').length },
  { id: 'needs_setup', label: 'Needs setup', count: all.value.filter(w => w.state === 'needs_setup').length },
  { id: 'other', label: 'Other', count: all.value.filter(w => w.kind === 'other').length },
])
const rows = computed(() => all.value.filter(w =>
  filter.value === 'all' ? true : filter.value === 'other' ? w.kind === 'other' : w.state === filter.value))

function dotFor(w) {
  if (w.in_progress) return 'b'
  if (w.state === 'ready') return 'g'
  return 'z'
}
function taskLabel(w) {
  const t = (w.task_types || [])[0]
  if (!t) return ''
  return t.replace(/-/g, ' ').replace(/^\w/, c => c.toUpperCase())
}
function subFor(w) {
  if (w.kind === 'other') return w.detail
  const parts = []
  const tl = taskLabel(w)
  if (tl) parts.push(tl)
  if (w.in_progress) parts.push('Setting up')
  else if (w.state === 'needs_setup' && w.summary) parts.push(w.summary)
  return parts.join(' · ')
}
async function rescan() { scanning.value = true; try { data.value = await api.rescan() } catch (e) { alert(e.message) } finally { scanning.value = false } }

async function openDetail(w) {
  detail.value = { slug: w.slug, name: w.name, state: w.state, loading: true }
  try { detail.value = await api.workflowDetail(w.slug) }
  catch (e) { detail.value = { slug: w.slug, name: w.name, state: w.state, error: e.message } }
}
function modelSub(m) {
  const parts = [m.folder || '']
  if (!m.installed) parts.push(m.no_source ? 'missing · no known source' : 'missing')
  if (m.gated) parts.push('gated')
  if (!m.installed && m.source) parts.push(m.source)
  return parts.filter(Boolean).join(' · ')
}
function openPlanFromDetail() {
  const d = detail.value
  detail.value = null
  openPlan({ slug: d.slug, name: d.name })
}

async function openPlan(w) {
  sheet.value = { w, loading: true }
  hfToken.value = ''
  for (const k of Object.keys(sources)) delete sources[k]
  try { const plan = await api.plan(w.slug); sheet.value = { w, plan } }
  catch (e) { sheet.value = { w, error: e.message } }
}
function closeSheet() { sheet.value = null }
const downloadsToDo = computed(() => (sheet.value?.plan?.downloads || []).filter(d => !d.already_present))
const packsToDo = computed(() => (sheet.value?.plan?.packs || []).filter(p => !p.installed))
const targetsLabel = computed(() => {
  const t = sheet.value?.plan?.targets || ['local']
  return t.map(x => x === 'local' ? 'this machine' : x).join(', ')
})
const canStart = computed(() => {
  const p = sheet.value?.plan
  if (!p) return false
  for (const b of p.blockers) {
    if (b.kind === 'hf_token' && !hfToken.value.trim() && !p.hf_token_set) return false
    if (b.kind === 'no_source' && !(sources[b.filename] || '').trim()) {
      // allow starting when there is *something* else to do
      if (downloadsToDo.value.filter(d => d.resolved).length === 0 && packsToDo.value.filter(x => x.installable).length === 0) return false
    }
  }
  return downloadsToDo.value.some(d => d.resolved || sources[d.filename]) || packsToDo.value.some(x => x.installable)
})
const startLabel = computed(() => {
  const dl = downloadsToDo.value.some(d => d.resolved || sources[d.filename])
  const inst = packsToDo.value.some(x => x.installable)
  return dl && inst ? 'Download & install' : dl ? 'Download' : inst ? 'Install' : 'Start'
})
async function start() {
  const w = sheet.value.w
  starting.value = true
  try {
    const src = {}
    for (const [k, v] of Object.entries(sources)) if (v && v.trim()) src[k] = v.trim()
    await api.setup(w.slug, { hf_token: hfToken.value.trim() || undefined, sources: Object.keys(src).length ? src : undefined })
    closeSheet()
    await load()
    emit('activity')
  } catch (e) { alert(e.message) } finally { starting.value = false }
}
</script>

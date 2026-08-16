<template>
  <div class="pills">
    <button v-for="f in filters" :key="f.id" class="pill" :class="{ on: filter === f.id }" @click="filter = f.id">{{ f.label }}<span class="n">{{ f.count }}</span></button>
  </div>
  <div class="body">
    <div v-if="!data" class="empty"><span class="spin"></span></div>
    <template v-else>
      <div class="grp" v-if="rows.length">
        <div v-for="w in rows" :key="w.key" class="li" :class="{ dim: w.kind === 'other' }">
          <span class="dot" :class="dotFor(w)"></span>
          <div class="t">
            <div class="a">{{ w.name }}</div>
            <div class="b">{{ subFor(w) }}</div>
          </div>
          <div class="r">
            <button v-if="w.state === 'needs_setup' && !w.in_progress" class="btn sm" @click="openPlan(w)">Get ready</button>
            <button v-else-if="w.in_progress" class="btn sm ghost" style="color:var(--accent-hi)" @click="$emit('activity')">Activity</button>
          </div>
        </div>
      </div>
      <div v-else class="empty">Nothing here.</div>
    </template>
  </div>
  <div class="foot">
    <span>{{ dirLabel }}</span>
    <span class="sp"></span>
    <button :disabled="scanning" @click="rescan">{{ scanning ? 'Scanning…' : 'Rescan' }}</button>
  </div>

  <div v-if="sheet" class="sheet-wrap" @click.self="closeSheet">
    <div class="sheet">
      <template v-if="sheet.loading">
        <h5>Checking what {{ sheet.w.name }} needs…</h5>
        <p><span class="spin"></span></p>
      </template>
      <template v-else-if="sheet.error">
        <h5>Couldn't plan setup</h5>
        <p>{{ sheet.error }}</p>
        <div class="acts"><button class="btn" @click="closeSheet">Close</button></div>
      </template>
      <template v-else>
        <h5>Set up {{ sheet.plan.name }}</h5>
        <p v-if="downloadsToDo.length">Downloads to {{ targetsLabel }} · {{ fmtBytes(sheet.plan.total_size) }}<span v-if="sheet.plan.free_space"> · {{ fmtBytes(sheet.plan.free_space) }} free</span></p>
        <p v-else-if="packsToDo.length">No downloads needed.</p>
        <div class="lst" v-if="downloadsToDo.length">
          <div v-for="d in downloadsToDo" :key="d.filename"><span class="mono f" :title="d.filename">{{ d.filename }}</span><span class="mono">{{ d.size ? fmtBytes(d.size) : (d.resolved ? '' : 'no source') }}</span></div>
        </div>
        <div class="lst" v-if="packsToDo.length">
          <div v-for="p in packsToDo" :key="p.url"><span class="f">Install {{ p.title }}</span><span class="mono">{{ p.installable ? 'node pack' : 'manual' }}</span></div>
        </div>

        <!-- blockers -->
        <div v-for="(b, i) in sheet.plan.blockers" :key="i" class="warn">
          <template v-if="b.kind === 'hf_token'">
            This needs a Hugging Face token. <a :href="b.license_url || 'https://huggingface.co/settings/tokens'" target="_blank" rel="noopener">Accept the model license ↗</a> if it has one, then paste a read token — it's saved on the ComfyUI machine for next time.
            <div class="field"><input v-model="hfToken" type="password" placeholder="hf_…" spellcheck="false" /></div>
          </template>
          <template v-else-if="b.kind === 'hf_license'">
            {{ b.repo || 'This model' }} requires accepting a license on Hugging Face. <a :href="b.license_url" target="_blank" rel="noopener">Accept it ↗</a>, then download.
          </template>
          <template v-else-if="b.kind === 'no_source'">
            No download source is known for <span class="mono">{{ b.filename }}</span>. Paste a direct URL to download it into <span class="mono">{{ b.folder || 'models' }}</span>, or install it yourself.
            <div class="field"><input v-model="sources[b.filename]" placeholder="https://…" spellcheck="false" /></div>
          </template>
          <template v-else-if="b.kind === 'unknown_node'">
            The node <span class="mono">{{ b.class_type }}</span> isn't in any known node pack. Install whichever pack provides it, then restart ComfyUI.
          </template>
          <template v-else-if="b.kind === 'no_manager'">
            ComfyUI-Manager isn't installed here, so node packs need a terminal on the ComfyUI machine:
            <div v-for="p in b.packs" :key="p.url" class="cmd mono">{{ p.manual }}</div>
            Restart ComfyUI afterwards; Stimma picks it up automatically.
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
const dirLabel = computed(() => 'Read from ComfyUI\'s workflows folder')

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
  if (w.in_progress) parts.push('Setting up — see Activity')
  else if (w.state === 'needs_setup' && w.summary) parts.push(w.summary)
  return parts.join(' · ')
}
async function rescan() { scanning.value = true; try { data.value = await api.rescan() } catch (e) { alert(e.message) } finally { scanning.value = false } }

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
  return dl && inst ? 'Download & install' : dl ? 'Download' : inst ? 'Install' : 'Nothing to do'
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

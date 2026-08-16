<template>
  <div class="body">
    <div v-if="!data" class="empty"><span class="spin"></span></div>
    <template v-else>
      <div class="grp" v-if="active.length">
        <h4>In progress</h4>
        <div v-for="o in active" :key="o.id" class="li">
          <span v-if="o.state === 'queued' || o.state === 'paused'" class="dot z"></span>
          <div class="t">
            <div class="a">{{ o.title }}</div>
            <div v-if="o.state === 'running' && o.kind === 'download'" class="bar" :class="{ ind: o.progress == null }"><i :style="{ width: (o.progress != null ? Math.round(o.progress * 100) : 40) + '%' }"></i></div>
            <div class="b" :style="o.state === 'running' && o.kind === 'download' ? 'margin-top:4px' : ''">{{ o.detail || labelFor(o.state) }}</div>
          </div>
          <div class="r">
            <button v-if="o.kind === 'download' && o.state === 'running'" class="btn sm ghost" title="Pause" @click="act(o, 'pause')">⏸</button>
            <button v-if="o.kind === 'download' && o.state === 'paused'" class="btn sm ghost" title="Resume" @click="act(o, 'retry')">▶</button>
            <button v-if="o.kind === 'download'" class="btn sm ghost" title="Cancel" @click="act(o, 'cancel')">✕</button>
          </div>
        </div>
      </div>
      <div class="grp" v-if="done.length">
        <h4>Done</h4>
        <div v-for="o in done" :key="o.id" class="li">
          <span class="dot" :class="o.state === 'done' ? 'g' : o.state === 'failed' ? 'r' : 'z'"></span>
          <div class="t">
            <div class="a">{{ o.title }}</div>
            <div class="b">{{ o.state === 'failed' ? (o.error || 'Failed') : (o.detail || labelFor(o.state)) }}<span v-if="o.finished_at"> · {{ fmtAgo(o.finished_at) }}</span></div>
          </div>
          <div class="r">
            <button v-if="o.state === 'failed'" class="btn sm" @click="fix(o)">Fix</button>
            <button v-else class="btn sm ghost" title="Remove" @click="act(o, 'dismiss')">✕</button>
          </div>
        </div>
      </div>
      <div v-if="!active.length && !done.length" class="empty">No activity</div>
    </template>
  </div>
  <div class="foot" v-if="done.length">
    <span class="sp"></span>
    <button @click="clearDone">Clear done</button>
  </div>

  <div v-if="fixing" class="sheet-wrap" @click.self="fixing = null">
    <div class="sheet">
      <h5>{{ fixing.title }}</h5>
      <p>{{ fixing.error }}</p>
      <template v-if="fixing.fix && fixing.fix.action === 'hf_license'">
        <p style="margin-top:6px"><a :href="fixing.fix.license_url" target="_blank" rel="noopener">Accept the license on Hugging Face ↗</a></p>
        <div class="acts"><button class="btn" @click="act(fixing, 'dismiss'); fixing = null">Remove</button><button class="btn primary" @click="act(fixing, 'retry'); fixing = null">Retry</button></div>
      </template>
      <template v-else-if="fixing.fix && fixing.fix.action === 'hf_token'">
        <p style="margin-top:6px"><a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Hugging Face read token ↗</a><template v-if="fixing.fix.license_url"> · <a :href="fixing.fix.license_url" target="_blank" rel="noopener">model license ↗</a></template></p>
        <div class="field"><input v-model="token" type="password" placeholder="hf_…" spellcheck="false" /></div>
        <div class="acts"><button class="btn" @click="act(fixing, 'dismiss'); fixing = null">Remove</button><button class="btn primary" :disabled="!token.trim()" @click="saveTokenAndRetry">Save & retry</button></div>
      </template>
      <template v-else-if="fixing.fix && fixing.fix.action === 'add_url'">
        <p style="margin-top:6px" class="mono">{{ fixing.meta.filename }}</p>
        <div class="field"><input v-model="url" placeholder="Download URL" spellcheck="false" /></div>
        <div class="acts"><button class="btn" @click="act(fixing, 'dismiss'); fixing = null">Remove</button><button class="btn primary" :disabled="!url.trim()" @click="redownload">Download</button></div>
      </template>
      <template v-else-if="fixing.fix && fixing.fix.action === 'manual'">
        <p style="margin-top:6px">On the ComfyUI machine, then restart:</p>
        <div class="cmd mono">{{ fixing.fix.command }}</div>
        <div class="acts"><button class="btn" @click="act(fixing, 'dismiss'); fixing = null">Done</button><button class="btn primary" @click="act(fixing, 'retry'); fixing = null">Retry</button></div>
      </template>
      <template v-else>
        <div class="acts"><button class="btn" @click="act(fixing, 'dismiss'); fixing = null">Remove</button><button class="btn primary" @click="act(fixing, 'retry'); fixing = null">Retry</button></div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api, fmtAgo } from './api'
const data = ref(null)
const fixing = ref(null)
const token = ref('')
const url = ref('')
let timer = null
async function load() { try { data.value = await api.activity() } catch (e) { /* keep last */ } }
onMounted(() => { load(); timer = setInterval(load, 1500) })
onUnmounted(() => clearInterval(timer))
const ops = computed(() => data.value?.operations || [])
const active = computed(() => ops.value.filter(o => !['done', 'failed', 'cancelled'].includes(o.state)))
const done = computed(() => ops.value.filter(o => ['done', 'failed', 'cancelled'].includes(o.state)))
function labelFor(s) { return { queued: 'Queued', running: 'Running', paused: 'Paused', done: 'Done', failed: 'Failed', cancelled: 'Cancelled' }[s] || s }
async function act(o, action) { try { await api.opAction(o.id, action); await load() } catch (e) { alert(e.message) } }
async function clearDone() { try { await api.clearDone(); await load() } catch (e) { alert(e.message) } }
function fix(o) { token.value = ''; url.value = ''; fixing.value = o }
async function saveTokenAndRetry() {
  try { await api.setCredentials({ huggingface_token: token.value.trim() }); await api.opAction(fixing.value.id, 'retry'); fixing.value = null; await load() } catch (e) { alert(e.message) }
}
async function redownload() {
  const o = fixing.value
  try {
    await api.addDownload({ filename: o.meta.filename, url: url.value.trim(), dest_path: o.meta.dest_path, group: o.group, workflows: o.meta.workflows, remember: true })
    await api.opAction(o.id, 'dismiss')
    fixing.value = null
    await load()
  } catch (e) { alert(e.message) }
}
</script>

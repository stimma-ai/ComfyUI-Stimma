<template>
  <div class="body">
    <div v-if="!s" class="empty"><span class="spin"></span></div>
    <template v-else>
      <div class="grp">
        <h4>Credentials</h4>
        <div class="li">
          <div class="t"><div class="a">Hugging Face token</div><div class="b">Gated models{{ srcNote(s.credentials.huggingface) }}</div></div>
          <div class="r"><span class="mono" v-if="s.credentials.huggingface.set">{{ s.credentials.huggingface.masked }}</span><button class="btn sm" @click="edit('huggingface_token')">{{ s.credentials.huggingface.set ? 'Change' : 'Add' }}</button></div>
        </div>
      </div>
      <div class="grp">
        <h4>ComfyUI</h4>
        <div class="li">
          <div class="t"><div class="a">ComfyUI-Manager</div><div class="b mono">{{ managerLabel }}</div></div>
          <div class="r">
            <button v-if="manager.state === 'missing' || manager.state === 'failed'" class="btn sm" :disabled="busy === 'manager'" @click="installManager">{{ busy === 'manager' ? 'Starting…' : manager.state === 'failed' ? 'Retry' : 'Install' }}</button>
            <span v-else-if="manager.state === 'installing'">Installing</span>
            <span v-else>Enabled</span>
          </div>
        </div>
        <div class="li">
          <div class="t"><div class="a">{{ s.instances.length > 1 ? 'Restart all instances' : 'Restart ComfyUI' }}</div></div>
          <div class="r"><button class="btn sm" :disabled="restarting" @click="restart">{{ restarting ? 'Restarting…' : 'Restart' }}</button></div>
        </div>
        <div class="li">
          <div class="t"><div class="a">Restore bundled workflows</div></div>
          <div class="r"><button class="btn sm" :disabled="busy === 'restore'" @click="restore">Restore</button></div>
        </div>
      </div>
      <div class="grp">
        <h4>ComfyUI-Stimma</h4>
        <div class="li">
          <div class="t"><div class="a">Update</div><div class="b mono">{{ updateLabel }}</div></div>
          <div class="r">
            <button v-if="upd && upd.update_available" class="btn sm primary" :disabled="busy === 'update'" @click="applyUpdate">{{ busy === 'update' ? 'Updating…' : 'Update & restart' }}</button>
            <button v-else class="btn sm" :disabled="busy === 'check'" @click="check">{{ busy === 'check' ? 'Checking…' : 'Check now' }}</button>
          </div>
        </div>
      </div>
    </template>
  </div>

  <div v-if="editing" class="sheet-wrap" @click.self="editing = null">
    <div class="sheet">
      <h5>Hugging Face token</h5>
      <p><a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens ↗</a></p>
      <div class="field"><input v-model="secret" type="password" placeholder="hf_…" spellcheck="false" /></div>
      <div class="acts"><button class="btn" @click="editing = null">Cancel</button><button class="btn primary" @click="saveSecret">Save</button></div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api } from './api'
const props = defineProps({ overview: Object, restarting: Boolean })
const emit = defineEmits(['refresh', 'restart', 'reconnecting'])
const s = ref(null)
const upd = ref(null)
const busy = ref('')
const editing = ref(null)
const secret = ref('')
let timer = null

async function load() { try { s.value = await api.settings(); upd.value = await api.updateStatus() } catch (e) { /* */ } }
onMounted(() => { load(); timer = setInterval(load, 3000) })
onUnmounted(() => clearInterval(timer))
function srcNote(c) { return c.set && c.source && c.source !== 'config' ? ` · from ${c.source}` : '' }
const updateLabel = computed(() => {
  const p = upd.value
  if (!p) return '…'
  if (!p.git) return 'Not a git checkout'
  if (p.error) return `${p.head || '?'} · ${p.error}`
  if (p.restart_required) return `${p.running_head} → ${p.head} · restart required`
  if (p.update_available) return [p.head, p.target].filter(Boolean).join(' → ')
  if (p.ahead) return `${p.head} · ahead of main`
  return `${p.head} · current`
})
const managerLabel = computed(() => {
  const m = manager.value
  if (m.state === 'failed') return m.operation?.error || 'Install failed'
  if (m.state === 'installing') return m.operation?.detail || 'Installing'
  if (m.state === 'restart_needed') return 'Restart required'
  return m.version || (m.installed ? 'Installed' : 'Not installed')
})
// The management UI can update before every running ComfyUI process has
// restarted into the matching Python backend. Keep Settings usable against
// the older `manager_present` response during that rolling-upgrade window.
const manager = computed(() => {
  if (s.value?.comfyui_manager) return s.value.comfyui_manager
  const installed = !!s.value?.manager_present
  return { state: installed ? 'ready' : 'missing', installed }
})
function edit(which) { editing.value = which; secret.value = '' }
async function saveSecret() { try { await api.setCredentials({ [editing.value]: secret.value }); editing.value = null; await load() } catch (e) { alert(e.message) } }
async function restart() {
  emit('restart')
}
async function restore() { busy.value = 'restore'; try { await api.restoreBundled() } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function check() { busy.value = 'check'; try { upd.value = await api.updateStatus(true) } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function applyUpdate() { busy.value = 'update'; try { const r = await api.updateApply(); if (!r.ok) alert(r.error); if (r.restarting) emit('reconnecting'); else await load(); emit('refresh') } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function installManager() { busy.value = 'manager'; try { await api.installManager(); await load(); emit('refresh') } catch (e) { alert(e.message) } finally { busy.value = '' } }
</script>

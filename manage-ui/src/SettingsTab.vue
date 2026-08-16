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
        <div class="li">
          <div class="t"><div class="a">Civitai API key</div><div class="b">Optional</div></div>
          <div class="r"><span class="mono" v-if="s.credentials.civitai.set">{{ s.credentials.civitai.masked }}</span><button class="btn sm" @click="edit('civitai_api_key')">{{ s.credentials.civitai.set ? 'Change' : 'Add' }}</button></div>
        </div>
      </div>
      <div class="grp">
        <h4>ComfyUI</h4>
        <div class="li">
          <div class="t"><div class="a">ComfyUI-Manager</div><div class="b mono">{{ managerLabel }}</div></div>
          <div class="r">
            <button v-if="s.comfyui_manager.state === 'missing' || s.comfyui_manager.state === 'failed'" class="btn sm" :disabled="busy === 'manager'" @click="installManager">{{ busy === 'manager' ? 'Starting…' : s.comfyui_manager.state === 'failed' ? 'Retry' : 'Install' }}</button>
            <span v-else-if="s.comfyui_manager.state === 'installing'">Installing</span>
            <span v-else>Enabled</span>
          </div>
        </div>
        <div class="li">
          <div class="t"><div class="a">{{ s.instances.length > 1 ? 'Restart all instances' : 'Restart ComfyUI' }}</div><div v-if="restartError" class="b error">{{ restartError }}</div></div>
          <div class="r"><button class="btn sm" :disabled="busy === 'restart'" @click="restart">{{ busy === 'restart' ? 'Restarting…' : 'Restart' }}</button></div>
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
            <button v-if="upd && upd.update_available" class="btn sm primary" :disabled="busy === 'update'" @click="applyUpdate">{{ busy === 'update' ? 'Updating…' : 'Update' }}</button>
            <button v-else class="btn sm" :disabled="busy === 'check'" @click="check">{{ busy === 'check' ? 'Checking…' : 'Check now' }}</button>
          </div>
        </div>
      </div>
    </template>
  </div>

  <div v-if="editing" class="sheet-wrap" @click.self="editing = null">
    <div class="sheet">
      <h5>{{ editing === 'huggingface_token' ? 'Hugging Face token' : 'Civitai API key' }}</h5>
      <p v-if="editing === 'huggingface_token'"><a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens ↗</a></p>
      <p v-else><a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account ↗</a></p>
      <div class="field"><input v-model="secret" type="password" :placeholder="editing === 'huggingface_token' ? 'hf_…' : 'key'" spellcheck="false" /></div>
      <div class="acts"><button class="btn" @click="editing = null">Cancel</button><button class="btn primary" @click="saveSecret">Save</button></div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api } from './api'
const props = defineProps({ overview: Object })
const emit = defineEmits(['refresh'])
const s = ref(null)
const upd = ref(null)
const busy = ref('')
const restartError = ref('')
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
  if (!p.git) return p.version
  if (p.error) return `${p.head || '?'} · ${p.error}`
  if (p.update_available) return `${p.head} · ${p.behind} behind`
  if (p.ahead) return `${p.head} · ahead of main`
  return `${p.head} · current`
})
const managerLabel = computed(() => {
  const m = s.value?.comfyui_manager
  if (!m) return '…'
  if (m.state === 'failed') return m.operation?.error || 'Install failed'
  if (m.state === 'installing') return m.operation?.detail || 'Installing'
  if (m.state === 'restart_needed') return 'Restart required'
  return m.version || (m.installed ? 'Installed' : 'Not installed')
})
function edit(which) { editing.value = which; secret.value = '' }
async function saveSecret() { try { await api.setCredentials({ [editing.value]: secret.value }); editing.value = null; await load() } catch (e) { alert(e.message) } }
async function restart() {
  busy.value = 'restart'
  restartError.value = ''
  try { await api.restart('all') }
  catch (e) { restartError.value = e.message; busy.value = '' }
}
async function restore() { busy.value = 'restore'; try { await api.restoreBundled() } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function check() { busy.value = 'check'; try { upd.value = await api.updateStatus(true) } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function applyUpdate() { busy.value = 'update'; try { const r = await api.updateApply(); if (!r.ok) alert(r.error); await load(); emit('refresh') } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function installManager() { busy.value = 'manager'; try { await api.installManager(); await load(); emit('refresh') } catch (e) { alert(e.message) } finally { busy.value = '' } }
</script>

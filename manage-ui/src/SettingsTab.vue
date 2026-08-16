<template>
  <div class="body">
    <div v-if="!s" class="empty"><span class="spin"></span></div>
    <template v-else>
      <div class="grp">
        <h4>Credentials</h4>
        <div class="li">
          <div class="t"><div class="a">Hugging Face token</div><div class="b">For gated models · stored on the ComfyUI machine{{ srcNote(s.credentials.huggingface) }}</div></div>
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
          <div class="t"><div class="a">{{ s.instances.length > 1 ? 'Restart all instances' : 'Restart ComfyUI' }}</div><div class="b">Applies node installs and plugin updates</div></div>
          <div class="r"><button class="btn sm" @click="restart">Restart</button></div>
        </div>
        <div class="li">
          <div class="t"><div class="a">Restore bundled workflows</div><div class="b">Re-copy Stimma reference workflows you've removed</div></div>
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
      <div class="grp">
        <h4>Diagnostics</h4>
        <div class="li" style="cursor:pointer" @click="toggleDiag('scan')"><div class="t"><div class="a">Workflow scan report</div></div><div class="r">{{ diag === 'scan' ? '▾' : '›' }}</div></div>
        <div v-if="diag === 'scan' && diagData" class="pre">{{ scanText }}</div>
        <div class="li" style="cursor:pointer" @click="toggleDiag('log')"><div class="t"><div class="a">Recent log</div></div><div class="r">{{ diag === 'log' ? '▾' : '›' }}</div></div>
        <div v-if="diag === 'log' && diagData" class="pre">{{ (diagData.log || []).slice(-80).join('\n') || 'Nothing logged yet.' }}</div>
      </div>
    </template>
  </div>

  <div v-if="editing" class="sheet-wrap" @click.self="editing = null">
    <div class="sheet">
      <h5>{{ editing === 'huggingface_token' ? 'Hugging Face token' : 'Civitai API key' }}</h5>
      <p v-if="editing === 'huggingface_token'">A read token from <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens ↗</a>. Saved to config.yaml on the ComfyUI machine. Leave empty to remove.</p>
      <p v-else>From <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account ↗</a>. Leave empty to remove.</p>
      <div class="field"><input v-model="secret" type="password" :placeholder="editing === 'huggingface_token' ? 'hf_…' : 'key'" spellcheck="false" /></div>
      <div class="acts"><button class="btn" @click="editing = null">Cancel</button><button class="btn primary" @click="saveSecret">Save</button></div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from './api'
const props = defineProps({ overview: Object })
const emit = defineEmits(['refresh'])
const s = ref(null)
const upd = ref(null)
const busy = ref('')
const editing = ref(null)
const secret = ref('')
const diag = ref('')
const diagData = ref(null)

async function load() { try { s.value = await api.settings(); upd.value = await api.updateStatus() } catch (e) { /* */ } }
onMounted(load)
function srcNote(c) { return c.set && c.source && c.source !== 'config' ? ` · from ${c.source}` : '' }
const updateLabel = computed(() => {
  const p = upd.value
  if (!p) return '…'
  if (p.error) return `${p.version} · ${p.error}`
  if (p.mode === 'dev') return `${p.branch || 'dev'} @ ${p.head || '?'}${p.behind ? ` · ${p.behind} commit${p.behind === 1 ? '' : 's'} behind` : ' · up to date'}`
  if (p.mode === 'static') return `${p.version} · not a git checkout`
  return p.update_available ? `${p.version} · ${p.latest} available` : `${p.version} · up to date`
})
function edit(which) { editing.value = which; secret.value = '' }
async function saveSecret() { try { await api.setCredentials({ [editing.value]: secret.value }); editing.value = null; await load() } catch (e) { alert(e.message) } }
async function restart() { if (!confirm('Restart ComfyUI now? Running jobs will be interrupted.')) return; try { await api.restart('all') } catch (e) { alert(e.message) } }
async function restore() { busy.value = 'restore'; try { await api.restoreBundled() } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function check() { busy.value = 'check'; try { upd.value = await api.updateStatus(true) } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function applyUpdate() { busy.value = 'update'; try { const r = await api.updateApply(); if (!r.ok) alert(r.error); await load(); emit('refresh') } catch (e) { alert(e.message) } finally { busy.value = '' } }
async function toggleDiag(which) { if (diag.value === which) { diag.value = ''; return } diag.value = which; try { diagData.value = await api.diagnostics() } catch (e) { diagData.value = { log: [String(e)] } } }
const scanText = computed(() => {
  const sc = diagData.value?.scan
  if (!sc) return ''
  const lines = []
  for (const d of sc.directories || []) lines.push(`dir  ${d}`)
  for (const t of sc.tools || []) { lines.push(`${t.warnings && t.warnings.length ? '⚠' : '✔'} ${t.slug}  ${t.file}`); for (const w of t.warnings || []) lines.push(`    ${w}`) }
  for (const o of sc.others || []) lines.push(`·  ${o.file}${o.has_stimma_nodes ? '  (Stimma nodes, no ToolInfo)' : ''}${o.error ? '  ' + o.error : ''}`)
  for (const d of sc.duplicates || []) lines.push(`dup ${d[0]}: kept ${d[1]} skipped ${d[2]}`)
  return lines.join('\n')
})
</script>

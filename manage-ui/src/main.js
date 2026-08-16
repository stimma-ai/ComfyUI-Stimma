import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

const params = new URLSearchParams(location.search)
const theme = params.get('theme') === 'light' ? 'light' : 'dark'
document.documentElement.setAttribute('data-theme', theme)
window.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'stimma-theme' && (e.data.theme === 'light' || e.data.theme === 'dark')) {
    document.documentElement.setAttribute('data-theme', e.data.theme)
  }
})

createApp(App).mount('#app')

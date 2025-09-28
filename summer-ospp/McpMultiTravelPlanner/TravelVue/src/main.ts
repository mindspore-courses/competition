import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router' // 导入路由配置

const app = createApp(App);
app.use(router) // 注册路由到应用
app.mount('#app') // 挂载到DOM
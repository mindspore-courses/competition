// src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
// 导入views中的子页面
import Chat from '../components/Chat.vue'
import Home from '../components/Home.vue'

// 路由规则：path（URL路径）对应component（子页面组件）
const routes = [
  {
    path: '/chat',      // 对话页路径（URL为xxx/chat）
    name: 'Chat',
    component: Chat // 关联ChatPage.vue子页面
  },
  {
    path: '/',          // 根路径重定向到对话页
    name: 'Home',
    component: Home
  },
]

// 创建路由实例
const router = createRouter({
  history: createWebHistory(), // 无#号的HTML5历史模式
  routes                       // 传入路由规则
})

export default router
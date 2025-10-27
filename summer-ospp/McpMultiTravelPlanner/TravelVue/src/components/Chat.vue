<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref } from 'vue';
import { marked } from 'marked';



const dynamicStep = ref('正在等待AI规划行程...'); // 用于动态显示当前步骤信息
const dynamicSteps = ref<string[]>(['']);

const getSSEMessage = async (requestData: undefined) => {

    try {
        const response = await fetch('http://127.0.0.1:5000/api/getRespone', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        dynamicStep.value = 'hahhaha';

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        if (!response.body) {
            throw new Error('响应体为空，无法读取SSE流');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        return {
            cancel: () => reader.cancel(),
            reader: (onMessage: (arg0: any) => void, onError: (arg0: unknown) => void, onComplete: () => void) => {
                const read = async () => {
                    try {
                        const { done, value } = await reader.read();
                        if (done) {
                            // 处理缓冲区中剩余的数据
                            if (buffer.trim()) {
                                const lines = buffer.split('\n');
                                for (const line of lines) {
                                    if (line.startsWith('data:')) {
                                        const data = line.slice(5).trim();
                                        latestStatus.value = data ? JSON.parse(data) : {};
                                        if (data) {
                                            try {
                                                onMessage(JSON.parse(data));
                                            } catch (e) {
                                                console.warn('解析最后一条消息失败:', e, '数据:', data);
                                            }
                                        }
                                    }
                                }
                            }
                            onComplete?.();
                            return;
                        }
                        
                        // 解码并追加到缓冲区
                        buffer += decoder.decode(value, { stream: true });
                        
                        // 按行分割并处理完整的消息
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || ''; // 最后一行可能不完整，保留在缓冲区
                        
                        for (const line of lines) {
                            if (line.startsWith('data:')) {
                                const data = line.slice(5).trim();
                                latestStatus.value = data ? JSON.parse(data) : {};
                                if (data) {
                                    try {
                                        
                                        onMessage(JSON.parse(data));
                                        
                                    } catch (e) {
                                        console.warn('解析消息失败:', e, '数据:', data);
                                        // 可以选择继续处理而不是抛出错误
                                    }
                                    console
                                }
                            }
                        }
                        
                        read(); // 继续读取
                    } catch (error) {
                        onError?.(error);
                    }
                };
                read();
            }
        };

    } catch (error) {
        console.error('SSE请求错误:', error);
        throw error;
    }
};

let isDone = ref(false); // 标记是否完成
let poi_name = ref(''); // 存储poi名称
let imgs = ref<string[]>([]); // 存储图片URL数组
let transportation = ref(''); // 存储交通方式

// 页面挂载后自动调用initChat
onMounted(async () => {
  // 读取保存的参数
    isDone.value = false;
    dynamicSteps.value.push("AI正在为您规划行程...");
    const savedData = localStorage.getItem('travelRequestData');
    
    if (savedData) {
        const requestData = JSON.parse(savedData);
        
        // 清除已保存的数据（避免重复调用）
        // localStorage.removeItem('travelRequestData');
        showUserRequirements(requestData);
        // 调用SSE接口
        const sse =  getSSEMessage(requestData);
        (await sse).reader(
            (parsedData) => { 
                // 这里的 parsedData 就是解析后的对象
               dynamicSteps.value[dynamicSteps.value.length - 1] = parsedData.message || "AI正在为您规划行程...";
                console.log("收到新消息：", parsedData.step, parsedData.message);
                // 可以在这里更新界面、存储数据等
                if('imgs' in parsedData){
                    console.log('收到图片数据:', parsedData);
                    isDone.value = true;
                    poi_name.value = parsedData.poi_name || '';
                    imgs.value = parsedData.imgs || [];
                    transportation.value = parsedData.transport || '';
                    dynamicSteps.value[dynamicSteps.value.length - 1] = parsedData.message+parsedData.more_message || "AI正在为您规划行程...";
                }
            },
            (error) => { 
                console.error("SSE错误：", error);
                showErrorMessage('与服务器的连接出现问题，请稍后重试。');
            },
            () => { 
                console.log("SSE连接已关闭");
                showLoadingState();
            }
        );
    } else {
        console.warn('未找到旅行请求参数');
        // 可以显示提示信息或默认界面
    }
});


// 定义旅行需求数据类型
interface TravelRequestData {
  city?: string;
  date?: string;
  people?: string;
  tag?: string;
  activityIntensity?: string;
  money?: string;
  food?: string;
}

// 定义消息类型
interface Message {
  id: number;
  type: 'user' | 'assistant' | 'system';
  content?: string;
  data?: TravelRequestData;
  isTravelRequest?: boolean;
  timestamp: string;
}

// 定义进度类型
interface ProcessStatus {
  step?: boolean | string | number | undefined; // 新增 boolean 类型
  message?: string | undefined;
  process?: number | undefined;
  content?: any; // 新增 content 字段（后端返回中包含，避免类型缺失）
}

// 响应式数据
const messages = ref<Message[]>([]);
const newMessage = ref('');
const chatContainer = ref<HTMLDivElement | null>(null);
// 响应式变量存储处理状态
const latestStatus = ref<ProcessStatus>({}); // 存储最新状态信息


// 生成当前时间字符串
const getCurrentTime = () => {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// 添加消息到列表并滚动到底部
const addMessage = (message: Omit<Message, 'id' | 'timestamp'>) => {
  const newMsg: Message = {
    ...message,
    id: Date.now(),
    timestamp: getCurrentTime()
  };
  
  messages.value.push(newMsg);
  
  // 滚动到底部
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
    }
  });
};

// 发送消息
const sendMessage = async () => {
  if (!newMessage.value.trim()) return;
  isDone.value = false;
  dynamicSteps.value.push(newMessage.value);
  dynamicSteps.value.push("AI正在为您规划行程...");
  // 添加用户消息
  addMessage({
    type: 'user',
    content: newMessage.value,
    isTravelRequest: false
  });

  const userMessage = newMessage.value;
  // 清空输入框
  newMessage.value = '';

  
  // 模拟助手回复（实际应用中会调用API）
  setTimeout(() => {
    addMessage({
      type: 'assistant',
      content: dynamicStep.value || '感谢您的消息，我们正在处理...',
      isTravelRequest: false
    });
  }, 800);
  
  const savedData = localStorage.getItem('travelRequestData');
  if (savedData) {
        const requestData = JSON.parse(savedData);
        requestData.prompt = userMessage;
        // 调用SSE接口
        const sse =  getSSEMessage(requestData);
        (await sse).reader(
            (parsedData) => { 
                // 这里的 parsedData 就是解析后的对象
                dynamicSteps.value[dynamicSteps.value.length - 1] = parsedData.message || "AI正在为您规划行程...";
                console.log("收到新消息：", parsedData.step, parsedData.message);
                // 可以在这里更新界面、存储数据等
                if('imgs' in parsedData){
                    console.log('收到图片数据:', parsedData);
                    isDone.value = true;
                    poi_name.value = parsedData.poi_name || '';
                    imgs.value = parsedData.imgs || [];
                    transportation.value = parsedData.transport || '';
                    dynamicSteps.value[dynamicSteps.value.length - 1] = parsedData.message+parsedData.more_message || "AI正在为您规划行程...";
                }
            },
            (error) => { 
                console.error("SSE错误：", error);
                showErrorMessage('与服务器的连接出现问题，请稍后重试。');
            },
            () => { 
                console.log("SSE连接已关闭");
                showLoadingState();
            }
        );
    } else {
        console.warn('未找到旅行请求参数');
        // 可以显示提示信息或默认界面
    }
};

// 显示用户旅行需求
const showUserRequirements = (data: TravelRequestData) => {
  addMessage({
    type: 'user',
    data,
    isTravelRequest: true
  });
  
  // 模拟助手回复
  setTimeout(() => {
    addMessage({
      type: 'assistant',
      content: dynamicStep.value || '感谢您的旅行需求，我们正在为您规划行程...',
      isTravelRequest: false
    });
  }, 1000);
};


// 更新聊天界面
function updateChatInterface(message: {
    message: string;
    step: boolean;
    process: undefined; content: any; 
}) {
    const chatContainer = document.getElementById('chat-container');
    const messageElement = document.createElement('div');
    messageElement.className = 'ai-message';
    messageElement.textContent = message.content || message;
    if(!chatContainer){
        console.error('未找到聊天容器元素');
        return;
    }
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 显示加载状态
function showLoadingState() {
    // const chatContainer = document.getElementById('chat-container');
    // const loadingElement = document.createElement('div');
    // loadingElement.id = 'loading-indicator';
    // loadingElement.className = 'loading';
    // loadingElement.textContent = 'AI正在为您规划行程...';
    // if(!chatContainer){
    //     console.error('未找到聊天容器元素');
    //     return;
    // }
    // chatContainer.appendChild(loadingElement);
}

// 显示错误信息
function showErrorMessage(message: string | null) {
    const chatContainer = document.getElementById('chat-container');
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.textContent = message;
    if(!chatContainer){
        console.error('未找到聊天容器元素');
        return;
    }
    chatContainer.appendChild(errorElement);
}

const renderedMarkdown = computed(() => {
  // 对dynamicStep的值进行Markdown转换
  return (idx: number) => {
    // 边界处理：确保索引有效
    if (idx < 0 || idx >= dynamicSteps.value.length) {
      return ''; // 或返回默认内容
    }
    // 根据索引获取对应内容并转换
    console.log(dynamicSteps)
    return marked.parse(dynamicSteps.value[idx].valueOf());
  };
});

</script>

<template>
    <div id="app-chat">
        <header>
            <div class="container header-content">
                <div class="logo">
                    <i class="fas fa-compass"></i>
                    <span>智能旅行伙伴</span>
                </div>
                <nav>
                    <ul>
                        <li><a href="/">首页</a></li>
                        <li><a href="#">使用教程</a></li>
                        <li><a href="#">关于我们</a></li>
                    </ul>
                </nav>
            </div>
        </header>
        <div id="chat-container">
            <!-- 聊天内容区 -->
            <div class="chat-messages" ref="chatContainer">
            <!-- 系统提示消息 -->
            <div class="message system">
                <p>请告诉我们您的旅行需求，我们将为您提供个性化建议</p>
            </div>
            
            <!-- 消息列表 -->
            <div 
                v-for="(msg, index) in messages" 
                :key="index" 
                :class="['message', msg.type]"
            >
                <div class="avatar">
                <span v-if="msg.type === 'user'">您</span>
                <span v-if="msg.type === 'assistant'">助</span>
                </div>
                <div class="content">
                <template v-if="msg.type === 'user' && msg.isTravelRequest">
                    <p><strong>我的旅行需求：</strong></p>
                    <p>📍 目的地：{{ msg.data?.city || '未选择' }}</p>
                    <p>📅 时间：{{ msg.data?.date || '未选择' }}</p>
                    <p>👥 人数：{{ msg.data?.people || '未选择' }}</p>
                    <p>🏷️ 标签：{{ msg.data?.tag || '未选择' }}</p>
                    <p>⚡ 活动强度：{{ msg.data?.activityIntensity || '未选择' }}</p>
                    <p>💰 预算：{{ msg.data?.money || '未选择' }}</p>
                    <p>🍽️ 饮食偏好：{{ msg.data?.food || '未选择' }}</p>
                </template>
                <template v-else>
                    <p v-html="renderedMarkdown(index)"></p>
                    <!-- <span v-if = "msg.type === 'assistant' && isDone === true">
                        <button>更多信息</button>
                        {{ poi_name }}, {{ imgs }}, {{ transportation }}
                    </span> -->
                </template>
                </div>
                <div class="timestamp">
                {{ msg.timestamp }}
                </div>
            </div>
            </div>
            
            <!-- 输入区域 -->
            <div class="chat-input-area">
            <textarea 
                v-model="newMessage" 
                placeholder="输入您的问题或需求..."
                @keyup.enter="sendMessage"
            ></textarea>
            <button @click="sendMessage">发送</button>
            </div>
        </div>
    </div>
</template>

<style scoped>
.progress-bar {
    width: 100%;
    height: 20px;
    background-color: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #45a049);
    transition: width 0.3s ease;
    border-radius: 10px;
}

.step-history {
    margin-top: 20px;
}

.step-item {
    padding: 8px;
    margin: 5px 0;
    border-left: 3px solid #4CAF50;
    background-color: #f9f9f9;
}





.chat-container {
  height: 620px;
  max-width: 800px;
  margin: 20px auto;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

.chat-header {
  background-color: #4f46e5;
  color: white;
  padding: 16px 20px;
  text-align: center;
}

.chat-header h2 {
  margin: 0 0 4px 0;
  font-size: 1.2rem;
}

.chat-header p {
  margin: 0;
  font-size: 0.9rem;
  opacity: 0.9;
}

.chat-messages {
  height: 600px;
  overflow-y: auto;
  padding: 20px;
  background-color: #f9fafb;
}

.message {
  margin-bottom: 16px;
  display: flex;
  max-width: 80%;
  animation: fadeIn 0.3s ease;
}

.message.system {
  max-width: 100%;
  justify-content: center;
}

.message.system .content {
  background-color: #e0e7ff;
  color: #3730a3;
  padding: 8px 16px;
  border-radius: 12px;
  font-size: 0.9rem;
}

.message.user {
  margin-left: auto;
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  color: white;
  margin: 0 8px;
  flex-shrink: 0;
}

.message.user .avatar {
  background-color: #4f46e5;
}

.message.assistant .avatar {
  background-color: #10b981;
}

.content {
  padding: 10px 16px;
  border-radius: 18px;
  line-height: 1.5;
}

.message.user .content {
  background-color: #4f46e5;
  color: white;
  border-bottom-right-radius: 4px;
}

.message.assistant .content {
  background-color: white;
  color: #1f2937;
  border: 1px solid #e5e7eb;
  border-bottom-left-radius: 4px;
}

.timestamp {
  font-size: 0.75rem;
  color: #9ca3af;
  align-self: flex-end;
  margin: 0 8px;
}

.chat-input-area {
  display: flex;
  padding: 12px;
  border-top: 1px solid #e5e7eb;
  background-color: white;
}

.chat-input-area textarea {
  flex-grow: 1;
  padding: 12px 16px;
  border: 1px solid #e5e7eb;
  border-radius: 24px;
  resize: none;
  outline: none;
  font-size: 1rem;
  min-height: 48px;
  max-height: 100px;
}

.chat-input-area textarea:focus {
  border-color: #4f46e5;
}

.chat-input-area button {
  margin-left: 12px;
  padding: 0 20px;
  background-color: #4f46e5;
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.chat-input-area button:hover {
  background-color: #4338ca;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

:root {
    --primary: #3a86ff;
    --secondary: #ff006e;
    --accent: #8338ec;
    --light: #f8f9fa;
    --dark: #212529;
    --success: #38b000;
    --warning: #ffbe0b;
    --info: #219ebc;
    --gray: #6c757d;
    --light-gray: #e9ecef;
}

body {
    background-color: #f5f7fa;
    color: var(--dark);
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    color: white;
    padding: 15px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
}

.logo i {
    margin-right: 10px;
}

nav ul {
    display: flex;
    list-style: none;
}

nav li {
    margin-left: 25px;
}

nav a {
    color: white;
    text-decoration: none;
    font-weight: 500;
    transition: opacity 0.3s;
}

nav a:hover {
    opacity: 0.8;
}

.hero {
    position: relative;
    height: 620px;
    overflow: hidden;
    border-radius: 0 0 15px 15px;
    margin-bottom: 40px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
}

.hero-background {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url('https://images.unsplash.com/photo-1469474968028-56623f02e42e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80') center/cover no-repeat;
    filter: brightness(0.85);
    transition: transform 10s ease;
}

.hero-content {
    position: relative;
    z-index: 2;
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    text-align: center;
    padding: 0 20px;
}

.hero h1 {
    font-size: 3rem;
    margin-bottom: 15px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.hero p {
    font-size: 1.2rem;
    max-width: 600px;
    margin-bottom: 30px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
}

.search-box {
    background: white;
    border-radius: 12px;
    padding: 25px;
    width: 90%;
    max-width: 800px;
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
}

.search-form {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: var(--dark);
}

.form-control {
    width: 100%;
    padding: 12px 15px;
    border: 2px solid var(--light-gray);
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.3s;
}

.form-control:focus {
    border-color: var(--primary);
    outline: none;
}

.counter {
    display: flex;
    align-items: center;
    border: 2px solid var(--light-gray);
    border-radius: 8px;
    overflow: hidden;
}

.counter button {
    background: var(--light-gray);
    border: none;
    padding: 12px;
    cursor: pointer;
    font-size: 1.2rem;
    transition: background 0.3s;
}

.counter button:hover {
    background: #dcdcdc;
}

.counter input {
    width: 50px;
    text-align: center;
    border: none;
    padding: 12px 5px;
    font-size: 1rem;
}

.btn {
    padding: 14px 25px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
}

.btn-primary {
    background: var(--primary);
    color: white;
}

.btn-primary:hover {
    background: #2563eb;
    transform: translateY(-2px);
}

.btn-link {
    background: transparent;
    color: var(--primary);
    text-decoration: underline;
}

.btn-link:hover {
    color: #2563eb;
}

.section-title {
    text-align: center;
    margin-bottom: 40px;
    color: var(--dark);
}

.section-title h2 {
    font-size: 2.2rem;
    margin-bottom: 15px;
    position: relative;
    display: inline-block;
}

.section-title h2:after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 4px;
    background: var(--primary);
    border-radius: 2px;
}

.section-title p {
    color: var(--gray);
    max-width: 700px;
    margin: 0 auto;
}

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 25px;
    margin-bottom: 50px;
}

.card {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

.card-img {
    height: 200px;
    overflow: hidden;
}

.card-img img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s;
}

.card:hover .card-img img {
    transform: scale(1.05);
}

.card-content {
    padding: 20px;
}

.card h3 {
    margin-bottom: 12px;
    color: var(--dark);
}

.card p {
    color: var(--gray);
    margin-bottom: 15px;
}

.preferences {
    background: white;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 40px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
}

.preferences-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.tag {
    padding: 8px 16px;
    background: var(--light-gray);
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s;
}

.tag:hover {
    background: #dee2e6;
}

.tag.selected {
    background: var(--primary);
    color: white;
}

.slider-container {
    padding: 10px 0;
}

.slider {
    -webkit-appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: var(--light-gray);
    outline: none;
}

.slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--primary);
    cursor: pointer;
}

.timeline {
    position: relative;
    max-width: 1000px;
    margin: 0 auto 50px;
}

.timeline::after {
    content: '';
    position: absolute;
    width: 6px;
    background-color: var(--light-gray);
    top: 0;
    bottom: 0;
    left: 50%;
    margin-left: -3px;
    border-radius: 3px;
}

.timeline-item {
    padding: 10px 40px;
    position: relative;
    width: 50%;
    box-sizing: border-box;
}

.timeline-item:nth-child(odd) {
    left: 0;
}

.timeline-item:nth-child(even) {
    left: 50%;
}

.timeline-content {
    padding: 20px;
    background-color: white;
    position: relative;
    border-radius: 12px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
}

.timeline-content h3 {
    margin-bottom: 10px;
    color: var(--dark);
}

.timeline-content p {
    margin-bottom: 15px;
    color: var(--gray);
}

.timeline-item::after {
    content: '';
    position: absolute;
    width: 25px;
    height: 25px;
    background-color: white;
    border: 4px solid var(--primary);
    border-radius: 50%;
    top: 20px;
    z-index: 1;
}

.timeline-item:nth-child(odd)::after {
    right: -13px;
}

.timeline-item:nth-child(even)::after {
    left: -13px;
}

.feedback-buttons {
    display: flex;
    gap: 10px;
    margin-top: 15px;
}

.feedback-btn {
    padding: 8px 15px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s;
}

.like-btn {
    background: #e8f5e9;
    color: var(--success);
}

.like-btn:hover {
    background: #c8e6c9;
}

.dislike-btn {
    background: #ffebee;
    color: var(--secondary);
}

.dislike-btn:hover {
    background: #ffcdd2;
}

footer {
    background: var(--dark);
    color: white;
    padding: 40px 0 20px;
}

.footer-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 30px;
    margin-bottom: 30px;
}

.footer-section h3 {
    margin-bottom: 20px;
    font-size: 1.3rem;
}

.footer-section ul {
    list-style: none;
}

.footer-section li {
    margin-bottom: 10px;
}

.footer-section a {
    color: #e0e0e0;
    text-decoration: none;
    transition: color 0.3s;
}

.footer-section a:hover {
    color: white;
}

.social-icons {
    display: flex;
    gap: 15px;
    margin-top: 20px;
}

.social-icons a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    transition: background 0.3s;
}

.social-icons a:hover {
    background: rgba(255, 255, 255, 0.2);
}

.copyright {
    text-align: center;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    color: #e0e0e0;
}

@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        text-align: center;
    }
    
    nav ul {
        margin-top: 15px;
    }
    
    nav li {
        margin: 0 10px;
    }
    
    .hero h1 {
        font-size: 2.2rem;
    }
    
    .hero p {
        font-size: 1rem;
    }
    
    .search-form {
        grid-template-columns: 1fr;
    }
    
    .timeline::after {
        left: 31px;
    }
    
    .timeline-item {
        width: 100%;
        padding-left: 70px;
        padding-right: 25px;
    }
    
    .timeline-item:nth-child(even) {
        left: 0;
    }
    
    .timeline-item::after {
        left: 18px;
    }
    
    .timeline-item:nth-child(odd)::after {
        right: auto;
    }
}
</style>

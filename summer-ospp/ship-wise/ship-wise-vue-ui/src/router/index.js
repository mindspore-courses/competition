import Vue from 'vue'
import Router from 'vue-router'

import Main from '../views/Main'
import Login from '../views/Login'
import Contact from '../views/Contact'
import Visualization from '../views/Visualization'
import More from '../views/more/More'
import Compress from '../views/more/Compress'

Vue.use(Router);

export default new Router({
	routes: [
		{
			//默认进入主页
			path: '/',
			component: Main,
			meta: {
				isLogin: false
			}
		}, {
			//登录页
			path: '/login',
			component: Login,
			meta: {
				isLogin: false
			}
		}, {
			//主页
			path: '/main',
			component: Main,
			meta: {
				isLogin: false
			}
		}, {
			//联系我们
			path: '/contact',
			component: Contact,
			meta: {
				isLogin: false
			}
		}, {
			//可视化面板
			path: '/Visualization',
			component: Visualization,
			meta: {
				isLogin: false
			}
		}, {
			//更多功能
			path: '/more',
			component: More,
			meta: {
				isLogin: false
			},
			children: [
				{
					//轨迹压缩
					path: '/more/compress',
					component: Compress,
					meta: {
						isLogin: false
					}
				}
			]
		},
		
	]
});
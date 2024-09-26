import Vue from 'vue'
import App from './App.vue'
import VueBus from 'vue-bus';
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import 'element-ui/lib/theme-chalk/display.css'
import store from './store'
import axios from 'axios'
import VueAxios from 'vue-axios'

import * as L from 'leaflet'
import 'leaflet/dist/leaflet.css'
Vue.use(router);
Vue.use(ElementUI);
Vue.use(store);
Vue.use(VueAxios, axios);
import { Popover } from 'element-ui';
Vue.use(Popover);
Vue.use(VueBus);

Vue.config.productionTip = false
Vue.L = Vue.prototype.$L = L

new Vue({
	router,
	store,
	render: h => h(App),
}).$mount('#app')

router.beforeEach((to, from, next) => {
	//获取用户登录成功后储存的登录标志
	let getFlag = localStorage.getItem("Flag");
	//判断用户是否已登录
	if(getFlag == 'isLogin'){
		store.state.isLogin = true;
		next();
		//若已登录再想进入登录界面，则直接定向回首页
		if(!to.meta.isLogin){
			next({
				path: '/main'
			})
		}
	}else{
		if(to.meta.isLogin){
			next({
				path: '/login',
			})
		}else{
			next();
		}
	}
});

router.afterEach(router => {
	window.scroll(0, 0);
});

/* Leaflet icon */
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
	iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
	iconUrl: require('leaflet/dist/images/marker-icon.png'),
	shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});



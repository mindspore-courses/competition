import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
	//设置属性
	state: {
		isLogin: false,
		userName: "",
	},
	//获取属性的状态
	getters: {
		isLogin: state => state.isLogin,
		userName: state => state.userName,
	},
	//设置属性状态
	mutations: {
		//保存登录状态
		userStatus(state, flag){
			state.isLogin = flag
		},
		//设置用户名
		setUserName(state, name){
			state.userName = name
		},
	},
	//应用mutations
	actions: {
		//获取登录状态
		userLogin({commit}, flag){
			commit("userStatus", flag)
		},
		//设置用户名
		setUserNameAction({commit}, name){
			commit("setUserName", name)
		},
	}
})
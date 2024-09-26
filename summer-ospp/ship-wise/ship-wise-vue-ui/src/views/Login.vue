<template>
	<div class="whole-container">
		<div class="whole-container-image"></div>
		<!-- 登录表单 -->
		<div class="login-container">
			<div class="login-title">MIPC账号登录</div>
			<el-form ref="loginForm" :model="form" :rules="rules" label-width="80px" class="login-box">
				<el-form-item label="账号" prop="username">
					<el-input type="text" placeholder="请输入账号" v-model="form.username" name="userName"/>
				</el-form-item>
				<el-form-item label="密码" prop="password">
					<el-input type="text" placeholder="请输入密码" v-model="form.password" name="userPass"/>
				</el-form-item>
				<el-button type="primary" @click="onSubmit('loginForm')" style="width: 91%;margin: 15px 0px 20px 30px;">登录</el-button>
			</el-form>
		</div>
		
		<!-- 弹出框 -->
		<el-dialog title="温馨提示" :visible.sync="dialogVisible" width="30%" :modal-append-to-body="false">
			<span>{{dialogInfo}}</span>
			<span slot="footer" class="dialog-footer">
				<el-button type="primary" @click="dialogVisible = false">确 定</el-button>
			</span>
		</el-dialog>
	</div>
</template>

<script>
export default {
	name: "Login",
	data() {
		return {
			form: {
				username: '',
				password: ''
			},

			// 表单验证，需要在 el-form-item 元素中增加 prop 属性
			rules: {
				username: [
					{required: true, message: '账号不可为空', trigger: 'blur'}
				],
				password: [
					{required: true, message: '密码不可为空', trigger: 'blur'}
				]
			},

			// 对话框显示和隐藏
			dialogVisible: false,
			dialogInfo: ''
		}
	},
	methods: {
		onSubmit(formName) {
			// 为表单绑定验证功能
			this.$refs[formName].validate((valid) => {
				if (valid) {
					//进行数据库验证
					var params = new URLSearchParams();
					params.append("userName", this.form.username);
					params.append("userPass", this.form.password);
					//前后端联调代码
					// this.axios.post(global.httpUrl + 'users/userLogin/', params)
					// .then(function(response){
					// 	if(response.data == "success"){
					// 		//存储登录状态
					// 		this.$store.dispatch("userLogin", true);
					// 		localStorage.setItem("Flag", "isLogin");
					// 		//存储用户名
					// 		this.$store.dispatch("setUserNameAction", this.form.username);
					// 		localStorage.setItem("userName", this.form.username);
							
					// 		//使用 vue-router 路由到指定页面，该方式称之为编程式导航
					// 		this.$router.push("/main");
					// 	}else{
					// 			this.dialogInfo = "用户名或密码输入错误";
					// 			this.dialogVisible = true;
					// 			this.form.username = "";
					// 			this.form.password = "";
					// 			return false;
					// 	}
					// }.bind(this))
					// .catch(function(error){
					// 	console.log(error);
					// });
					
					//前端本地测试代码
					this.axios.get('/user.json')
					.then(function(response) {
						if(response.data.loginSuccess == "yes"){      //验证成功
							//存储登录状态
							this.$store.dispatch("userLogin", true);
							localStorage.setItem("Flag", "isLogin");
							//存储用户名
							this.$store.dispatch("setUserNameAction", this.form.username);
							localStorage.setItem("userName", this.form.username);
							
							//使用 vue-router 路由到指定页面，该方式称之为编程式导航
							this.$router.push("/main");
						}else{      //验证失败
							this.dialogInfo = "用户名或密码输入错误";
							this.dialogVisible = true;
							this.form.username = "";
							this.form.password = "";
							return false;
						}
					}.bind(this));
					
				} else {
					this.dialogInfo = "请输入账号和密码";
					this.dialogVisible = true;
					return false;
				}
			});
		}
	}
}
</script>

<style lang="scss" scoped>
.whole-container {
	position: fixed;
	width: 100%;
	height: 100%;
	top: 0px;
	left: 0px;
}
.whole-container-image {
	position: fixed;
	// position: absolute;
	width: 100%;
	height: 100%;
	top: 0px;
	left: 0px;
	background-image: url("../assets/login-bg.jpg");
	transform: translate3d(0px, 0px, 0px);
	background-size: cover;
	background-position: center center;
	z-index: -1;
	filter: brightness(40%);
	-webkit-filter: brightness(40%);
}
.login-container {
	background-color: #FFFFFF;
	border: 1px solid #DCDFE6;
	width: 360px;
	height: 310px;
	margin: auto;
	position: absolute;
	top: 0;
	left: 0;
	right: 0;
	bottom: 0;
	border-radius: 8px;
	-webkit-border-radius: 8px;
	-moz-border-radius: 8px;
	box-shadow: 0 0 25px #909399;
}

.login-title {
	text-align: center;
	margin: 0 auto 40px auto;
	color: #fff;
	background-color: #5297f8;
	padding-top: 15px;
	padding-bottom: 15px;
	font-size: 16px;
}
.login-box {
	padding-right: 35px;
}

</style>

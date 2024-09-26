<template>
	<div>
		<el-container>
			<!-- 导航栏 -->
			<el-header style="padding: 0;">
				<el-menu
					:default-active="this.$route.path"
					router mode="horizontal"
					background-color="#18223d"
					text-color="#fff"
					active-text-color="#ffd04b">
						<el-menu-item v-for="(item, i) in navList" :key="i" :index="item.name">
							<template slot="title">
								<i :class="item.iconUrl"></i>
								<span>{{item.navItem}}</span>
							</template>
						</el-menu-item>
					</el-menu>
			</el-header>
			<!-- 主体 -->
			<el-main style="padding: 0">
				<router-view></router-view>
			</el-main>
		</el-container>
	</div>
</template>

<script>
export default {
	name: 'More',
	data() {
		return {
			navList: [
				{name: '/main', navItem: '回到首页', iconUrl: 'el-icon-s-home'},
				{name: '/more/compress', navItem: '轨迹压缩', iconUrl: 'el-icon-s-platform'},
				{name: '/more/func2', navItem: '功能2', iconUrl: 'el-icon-s-platform'},
				{name: '/more/func3', navItem: '功能3', iconUrl: 'el-icon-s-platform'}
			]
		}
	},
	methods: {
		
	}
}
</script>

<style>
* {
	margin: 0;
	padding: 0;
}
</style>

<template>
  <div>
    <el-container>
      <!-- 主体部分 -->
      <el-main style="padding: 0;">
        <!-- 走马灯 -->
        <el-carousel :interval="4000" height="400px">
          <el-carousel-item v-for="item in imgList" :key="item.id">
            <img :src="item.idView" class="carousel_img"/>
          </el-carousel-item>
        </el-carousel>
        <div style="padding: 20px; background: #fff">
          <!-- 关于我们 -->
          <div class="aboutus">
            <div class="aboutus_info">
              <h2>关于我们</h2>
              <div class="bottom_block"></div>
              <p>我们专注于海事数据挖掘服务！</p>
              <p>
                我们推出基于AIS的船位信息服务，针对海事数据挖掘领域的各项成果和最新算法进行实现，并对外提供相应的可视化服务。我们的目标是形成完整的集陆地、海洋、卫星于一体的船舶AIS监控网络和可视化服务。
              </p>
              <p>
                我们将不懈努力，以信息技术提升海运数据挖掘的效率。
              </p>
            </div>
          </div>
          <!-- 联系我们 -->
          <div class="w_1226">
            <div class="contactus">
              <h2>联系我们</h2>
              <div class="bottom_block"></div>
              <ul>
                <li>
                  <el-card class="box-card">
                    <div slot="header" class="clearfix">
                      <span>服务咨询</span>
                    </div>
                    <div class="card-content">
                      <p class="c_999">船舶轨迹大数据可视化系统相关业务咨询，账号注册，功能介绍等</p>
                    </div>
                  </el-card>
                </li>
                <li class="box-card-li">
                  <el-card class="box-card">
                    <div slot="header" class="clearfix">
                      <span>轨迹数据</span>
                    </div>
                    <div class="card-content">
                      <p class="c_999">AIS船位，历史轨迹，详细船舶档案</p>
                    </div>
                  </el-card>
                </li>
                <li class="box-card-li">
                  <el-card class="box-card">
                    <div slot="header" class="clearfix">
                      <span>API服务</span>
                    </div>
                    <div class="card-content">
                      <p class="c_999">船位API数据接口，项目定制化开发，热力图，航运数据分析等</p>
                    </div>
                  </el-card>
                </li>
              </ul>
            </div>
          </div>
          <!-- 地址 -->
          <div class="address">
            <p class="address_info">
              地址：海南省三亚市崖州区新道街9号创新研学谷
            </p>
          </div>
          <!-- 地图 -->
          <div class="contact-map">
            <img src="../assets/address.png"/>
          </div>
          <!-- 返回首页 -->
          <div class="backtoindex" @click="skipToIndex">
            <span><i class="el-icon-back" style="font-size: 22px;"></i></span>
            <span style="margin-left: 15px;">返回首页</span>
          </div>
        </div>
      </el-main>
      <!-- 底栏 -->
      <el-footer style="background-color: #201f25;margin-top: 60px;color: #fff;">
        <div class="footer-message">
					<span>
						<i class="el-icon-message"></i>
						邮箱：coderjiang@whut.edu.cn
					</span>
          <span class="footer-message-span">
						<i class="el-icon-mobile-phone"></i>
						电话：190 1149 0549
					</span>
          <span class="footer-message-span">
						<i class="el-icon-document"></i>
						ShipWise
					</span>
        </div>
      </el-footer>
      <!-- 回到顶部 -->
      <div>
        <el-backtop></el-backtop>
      </div>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'Contact',
  data() {
    return {
      imgList: [
        {id: 0, idView: require("../assets/school1.jpg")},
        {id: 1, idView: require("../assets/school2.jpg")},
        {id: 2, idView: require("../assets/school3.jpg")},
        {id: 3, idView: require("../assets/school4.jpg")},
      ],       //走马灯图片
    };
  },
  methods: {
    //跳转到首页
    skipToIndex() {
      this.$router.push("/main");
    }
  },
  mounted() {
  }
}
</script>

<style>
.carousel_img {
  width: 100%;
  height: inherit;
}

.aboutus {
  position: relative;
  /* margin-top: -50px; */
  padding-bottom: 20px;
}

.aboutus_info {
  background-color: rgba(255, 255, 255, 0.96);
  width: 1226px;
  margin: 0 auto;
  font-size: 14px;
  color: #534C4A;
  line-height: 28px;
}

.aboutus_info h2 {
  font-size: 26px;
  color: #141313;
  padding: 40px 0 20px;
}

.bottom_block {
  width: 40px;
  height: 5px;
  border-radius: 5px;
  background: #E93E27;
  margin-bottom: 20px;
}

.w_1226 {
  width: 1226px;
  height: 400px;
  margin: 0 auto;
  background-color: rgba(255, 255, 255, 0.96);
}

.contactus h2 {
  font-size: 26px;
  color: #141313;
  padding: 40px 0 20px;
}

h1, h2, h3, h4, h5, h6 {
  font-weight: 400;
}

body, div, h1, h2, h3, h4, h5, html, input, li, p, td, th, ul {
  font-family: "微软雅黑";
}

div, li, p, ul {
  display: block;
}

.contactus li {
  display: inline-block;
  float: left;
  width: 350px;
  background: rgba(255, 255, 255, 0.96);
  height: 260px;
}

.box-card-li {
  margin-left: 88px;
}

.clearfix span {
  font-size: 28px;
  font-weight: normal;
  margin-top: 10px;
  color: #666;
}

.card-content p {
  margin-top: 20px;
}

.c_999 {
  color: #999;
}

.box-card {
  height: 260px;
}

.address, .contact-map, .backtoindex {
  width: 1226px;
  margin: 0 auto;
  background-color: rgba(255, 255, 255, 0.96);
}

.address .address_info {
  line-height: 80px;
}

.contact-map img {
  width: inherit;
  border: none;
  vertical-align: middle;
  display: inline-block;
}

.footer-message {
  width: 970px;
  line-height: 60px;
  margin: 0 auto;
}

.footer-message-span {
  margin-left: 125px;
}

.backtoindex {
  text-align: center;
  margin-top: 60px;
  font-size: 22px;
}

.backtoindex:hover {
  cursor: pointer;
  color: red;
}

</style>

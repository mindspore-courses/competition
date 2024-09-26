<template>
  <div class="All-Body">
    <el-container>
      <!-- 导航栏 -->
      <el-header style="background-color: rgba(0, 85, 255, 1);height: calc(7vh);">
        <!-- 此处写导航栏代码 -->
        <div>
          <el-row type="flex">
            <el-col :span="20" class="hidden-md-and-down">
              <el-menu class="el-menu-traj" mode="horizontal" background-color="rgba(0, 85, 255, 1)"
                       text-color="#fffefd" active-text-color="#ffffff" style="border-bottom: none;">
                <el-row type="flex">
                  <!-- 标题 -->
                  <el-col :span="7">
                    <el-menu-item>
                      <div class="title-span">船舶轨迹大数据可视化系统</div>
                    </el-menu-item>
                  </el-col>
                  <!-- 菜单 -->
                  <!-- 虚拟交通网络 -->
                  <el-col :span="1.5">
                    <el-menu-item>
                      <div class="func-span" @click="virtualTransNetwork">AIS数据</div>
                    </el-menu-item>
                  </el-col>
                  <!-- 轨迹聚类 -->
                  <el-col :span="1.5">
                    <el-menu-item>
                      <div class="func-span">卫星图像</div>
                    </el-menu-item>
                  </el-col>
                  <!-- 可视化面板 -->
                  <el-col :span="1.5">
                    <el-menu-item>
                      <div class="func-span" @click="skipToVisual">可视化面板</div>
                    </el-menu-item>
                  </el-col>
                  <!-- 联系我们 -->
                  <el-col :span="1.5">
                    <el-menu-item>
                      <div class="func-span" @click="skipToContact">联系我们</div>
                    </el-menu-item>
                  </el-col>
                  <!-- 更多功能 -->
                  <el-col :span="1.5">
                    <el-menu-item>
                      <el-dropdown @command="skipToMore">
                        <div class="func-span el-dropdown-link">
                          更多功能<i class="el-icon-arrow-down el-icon--right"></i>
                        </div>
                        <el-dropdown-menu slot="dropdown">
                          <el-dropdown-item icon="el-icon-circle-check" command="compress">
                            轨迹压缩
                          </el-dropdown-item>
                        </el-dropdown-menu>
                      </el-dropdown>
                    </el-menu-item>
                  </el-col>
                </el-row>
              </el-menu>
            </el-col>
            <!-- 菜单图标 -->
            <el-col :xs="9" class="hidden-md-and-up">
              <div class="nav-icon" @click="drawer = true">
                <i class="el-icon-s-unfold" style="font-size: 30px;"></i>
              </div>
            </el-col>
            <!-- 登录 -->
            <el-col :span="4" :xs="15">
              <div class="rightsection">
                <div v-if="$store.state.isLogin == true">
                  欢迎您，{{ $store.state.userName }}
                  <el-button type="primary" @click="logout" style="margin-left: 5px;">退出</el-button>
                </div>
                <span class="btn-click" v-else @click="skipToLogin">登录</span>
              </div>
            </el-col>
          </el-row>
        </div>
      </el-header>

      <!-- 添加新布局 -->
      <el-container id="Main">
        <el-aside :width="isCollapse ? '0px' : '350px'" style="display:{dp}">

          <!-- <div class="toggle-button" @click="toggleCollapse()">|||</div> -->
          <el-aside style="height: 470px;width: auto;">
            <image-list @change="handleChangeShip" ref="boat_id" @picture_list="pic_list" @picture_list1="pic_list1">
            </image-list>
          </el-aside>
          <el-divider></el-divider>
          <el-aside style="height: 350px;width: auto;">
            <!-- 新添功能：船只信息（左下角） -->
            <boat-list></boat-list>
          </el-aside>
        </el-aside>

        <el-main style="padding: 0em;">
          <span id="left" @click="toggleCollapse()" v-show="index !== 0">&lt;</span>
          <span id="right" @click="toggleCollapse1()" v-show="index !== 0">&gt;</span>
          <map-list></map-list>
        </el-main>

        <el-aside :width="isCollapse1 ? '0px' : '350px'">
          <!-- 右上角新添表格 -->
          <el-aside style="height: 470px;width: auto;">
            <table-list></table-list>
          </el-aside>
          <el-divider></el-divider>
          <!-- 右下角 -->
          <el-aside style="height: 450px;width: auto;">
            <SecondaryImage></SecondaryImage>
            <!-- <el-button @click="changeimage()">切换图片</el-button> -->
          </el-aside>
        </el-aside>
      </el-container>
    </el-container>

    <!-- 抽屉 -->
    <!-- <el-drawer title="船舶轨迹大数据可视化系统" :visible.sync="drawer" :direction="direction" size="45%"
      class="hidden-md-and-up">
      <el-menu>
        <el-menu-item @click="virtualTransNetwork">
          <i class="el-icon-menu"></i>
          虚拟交通网络
        </el-menu-item>
      </el-menu>
      <el-menu>
        <el-menu-item @click="drawer = false">
          <i class="el-icon-menu"></i>
          轨迹聚类
        </el-menu-item>
      </el-menu>
      <el-menu>
        <el-menu-item @click="skipToVisual">
          <i class="el-icon-menu"></i>
          可视化面板
        </el-menu-item>
      </el-menu>
      <el-menu>
        <el-menu-item @click="skipToContact">
          <i class="el-icon-menu"></i>
          联系我们
        </el-menu-item>
      </el-menu>
    </el-drawer>
 -->
  </div>
</template>

<script>
import global from './Common.vue'     // 引入全局文件
import MiniMap from 'leaflet-minimap'    // 引入鹰眼图组件
import 'leaflet-minimap/dist/Control.MiniMap.min.css'
import Fullscreen from 'leaflet-fullscreen'     // 引入全屏组件
import 'leaflet-fullscreen/dist/leaflet.fullscreen.css'
import {Message, MessageBox, Result} from 'element-ui'
import 'leaflet-draw'    // 引入绘图组件
import 'leaflet-draw/dist/leaflet.draw.css'
import * as echarts from "echarts";
import ImageList from '../components/ImageList.vue'
import TableList from '../components/Tablelist.vue'
import BoatList from '../components/BoatList.vue'
import MapList from '../components/MapList.vue'
import axios from 'axios'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import SecondaryImage from '../components/SecondaryImage.vue'
import {getDetections, getPoints} from "@/assets/js/api/common-api";


export default {
  name: "Main",
  components: {
    "image-list": ImageList,
    "table-list": TableList,
    "boat-list": BoatList,
    "map-list": MapList,
    SecondaryImage
  },
  data() {
    return {
      map: "",     //地图
      trajIcon: "",     //图标
      layerGroup1: null,
      layerGroup: null,    //船舶图层组
      // point: ""
      tianditu_1_tile: "",     //天地图矢量底图
      tianditu_1_marker: "",      //天地图矢量标记
      tianditu_2_tile: "",     //天地图影像底图
      tianditu_2_marker: "",      //天地图影像标记
      tianditu_3_tile: "",     //天地图地形底图
      tianditu_3_marker: "",      //天地图地形标记
      basicLayer: "",       //底图图层
      markerLayer: "",      //注解图层
      loading: true,        //加载动画
      drawer: false,        //抽屉显示
      direction: "ltr",     //抽屉方向
      trajMMSI: "",      //船舶MMSI
      trajSign: "",      //船舶呼号
      trajIMO: "",       //船舶IMO
      trajVesselType: "", //船舶类型
      trajVesselLength: "",   //船舶船长
      trajVesselWidth: "",    //船舶船宽
      trajDeepMax: "",    //船舶吃水
      trajHeading: "",   //船舶船首向
      trajCourse: "",    //船舶船迹向
      trajSpeed: "",     //船舶航速
      trajLat: "",       //船舶纬度
      trajLon: "",       //船舶经度
      trajTarget: "",    //船舶目的地
      trajTimeStamp: "", //船舶更新时间
      trajVesselName: "",     //船舶船名
      countryPic: "",    //国家图片
      txtShow: false,    //搜索框的x是否可见
      contShow: false,   //内容框是否可见
      mapLon: "",     //经度
      mapLat: "",     //纬度
      zoomLevel: "",  //缩放级别
      shipNum: "",    //船舶数量
      centerIcon: "",     //中心点图标
      drawControl: null,    //画图控件
      drawLayerGroup: null,    //图形图层组
      drawObj: null,     //绘制对象
      drawType: null,    //绘制类型
      myChartStyle: {float: "left", width: "100%", height: "340px"},//图表样式
      image_id: [],
      img_list: [], //所有大图的列表
      shipPath: 'ship0.jpg',
      imagenumber: 0,
      imageSet: [{
        url: 'ship1.jpg'
      }, {
        url: 'ship2.jpg'
      }],
      isCollapse: true,//隐藏右侧标记
      isCollapse1: true, //隐藏左侧标记
    };
  },
  // mounted() {
  // 	setTimeout(() => this.initEcharts(), 250);
  // },


  //将imageList中获取的数据传到boatList中
  methods: {
    //隐藏右侧内容
    toggleCollapse1() {
      this.isCollapse1 = !this.isCollapse1;
      this.zoomLevel = this.map.getZoom();
    },

    //隐藏左侧内容
    toggleCollapse() {
      this.isCollapse = !this.isCollapse;
      this.zoomLevel = this.map.getZoom();
    },

    //改变子图图像
    changeimage() {
      if (this.imagenumber === 0) {
        this.shipPath === this.imageSet[0].url;
        this.imagenumber = 1;
      } else if (this.imagenumber === 1) {
        this.shipPath = this.imageSet[1].url;
        this.imagenumber = 0;
      }
    },

    //获取接口中所有的大图列表
    async pic_list() {
      let img_list = []
      let {data} = await getDetections()
      for (let item of data) {
        if (!img_list.includes(item[0])) {
          img_list.push(item[0])
        }
      }
      // const json_files = require.context("../public/json", true, /\.json$/).keys();
      //将大图列表传给ImageList组件
      this.$refs.boat_id.update1(img_list)
    },

    async pic_list1(queryInfo) {
      let img_list1 = []
      let {data} = await getDetections(queryInfo.createTimeFrom, queryInfo.createTimeTo)
      for (let item of data) {
        if (!img_list1.includes(item[0])) {
          img_list1.push(item[0])
        }
      }
      // const json_files = require.context("../public/json", true, /\.json$/).keys();
      //将大图列表传给ImageList组件
      this.$refs.boat_id.update1(img_list1)
    },


    //地图标点
    clickTr(row, event, column) {
      //console.log(row["id"]) 跟下面效果一样
      var a = row["name"].split(",")[0];
      var b = row["name"].split(",")[1];
      this.map.panTo([b, a]);
      alert(row["name"]);//获取各行id的值
    },

    handleSelectionChange(val) {

      this.selectIndex = val;
    },

    handleDoubleClick(row, column, event) {
      console.log(row);
      this.$message('加载第' + row.name + "行");
    },

    shipDetection() {
      console.log(this.selectIndex)
    },


    //跳转到登录界面
    skipToLogin() {
      this.$router.push("/login");
    },
    //跳转到可视化面板
    skipToVisual() {
      this.drawer = false;     //关闭抽屉
      this.$router.push("/visualization");
    },
    //跳转到联系我们
    skipToContact() {
      this.drawer = false;     //关闭抽屉
      this.$router.push("/contact");
    },
    //跳转到轨迹压缩
    skipToMore(command) {
      this.drawer = false;     //关闭抽屉
      if (command == "compress") {
        this.$router.push("/more/compress");
      }

    },
    //注销
    logout() {
      localStorage.removeItem("Flag");
      this.$store.dispatch("userLogin", false);
    },
    //清空搜索框
    clearInputTxt() {
      document.getElementById("txtKey").value = "";
      this.txtShow = false;
    },
    //查看内容框时间
    readTime() {
      this.$notify({
        title: '更新时间',
        type: 'success',
        message: this.trajTimeStamp
      });
    },
    // 搜索船舶信息
    searchShip() {
      let txtKey = document.getElementById("txtKey").value;
      let that = this;
      let params = new URLSearchParams();
      params.append("txtKey", txtKey);
      this.axios.post(global.httpUrl + 'points/searchPoint/', params)
          .then(function (response) {
            if (response.data.aisInfo.length === 0) {    //搜索不到内容
              that.contShow = false;
              Message({
                message: "搜索不到内容",
                type: "error"
              });
              return;
            }
            // 设置内容框内容
            this.contShow = true;
            that.trajMMSI = (response.data.aisInfo)[0].fields.mmsi;
            that.trajSign = (response.data.aisInfo)[0].fields.sign;
            that.trajIMO = (response.data.aisInfo)[0].fields.IMO;
            that.trajVesselType = global.transVesselType((response.data.aisInfo)[0].fields.vesselType);
            that.trajVesselLength = (response.data.aisInfo)[0].fields.vesselLength;
            that.trajVesselWidth = (response.data.aisInfo)[0].fields.vesselWidth;
            that.trajDeepMax = (response.data.aisInfo)[0].fields.deepMax;
            that.trajHeading = (response.data.aisInfo)[0].fields.heading;
            that.trajCourse = (response.data.aisInfo)[0].fields.course;
            that.trajSpeed = (response.data.aisInfo)[0].fields.speed;
            that.trajLat = (response.data.aisInfo)[0].fields.latitude;
            that.trajLon = (response.data.aisInfo)[0].fields.longitude;
            that.trajTarget = (response.data.aisInfo)[0].fields.target;
            that.trajTimeStamp = (response.data.aisInfo)[0].fields.timestamp;
            that.trajVesselName = (response.data.aisInfo)[0].fields.vesselName;
            that.countryPic = global.transCountry((response.data.aisInfo)[0].fields.countryName);
          }.bind(this))
          .catch(function (error) {
            console.log(error);
          });
    },
    //发送建议
    pushAdvice() {
      var that = this;
      if (!this.$store.state.isLogin) {
        this.$confirm("请先进行登录！", "警告", {
          confirmButtonText: '去登录',
          cancelButtonText: '取消',
          type: 'warning'
        }).then(() => {
          that.$router.push("/login");
        });
        return;
      }
      const h = this.$createElement;
      this.$msgbox({
        title: "意见反馈",
        message: h('div', {
          attrs: {
            class: "leaflet_msg"
          }
        }, [
          h('div', null, [
            h('ul', {
              attrs: {
                class: "leave_message"
              }
            }, [
              h('li', null, [
                h('span', null, "电子邮箱"),
                h('input', {
                  attrs: {
                    type: "text",
                    id: "yourEmail",
                    placeholder: "您的电子邮箱"
                  }
                }, null)
              ]),
              h('li', null, [
                h('span', null, "手机号码"),
                h('input', {
                  attrs: {
                    type: "text",
                    id: "yourMobile",
                    placeholder: "您的联系电话"
                  }
                }, null)
              ]),
              h('li', null, [
                h('span', {
                  attrs: {
                    style: "vertical-align:top"
                  }
                }, "建议内容"),
                h('textarea', {
                  attrs: {
                    type: "text",
                    id: "messageInfo"
                  }
                }, null)
              ]),
            ])
          ])
        ]),
        showCancelButton: true,
        confirmButtonText: "提交",
        cancelButtonText: "取消"
      }).then(() => {
        that.pushFeedback();
      }).catch(() => {
        Message.info({
          type: 'info',
          message: "取消输入"
        });
      });
    },
    //推送意见反馈
    pushFeedback() {
      var params = new URLSearchParams();
      params.append("email", document.getElementById("yourEmail").value);
      params.append("phone", document.getElementById("yourMobile").value);
      params.append("content", document.getElementById("messageInfo").value);
      params.append("userName", this.$store.state.userName);
      this.axios.post(global.httpUrl + '/userFeedback/', params)
          .then(function (response) {
            let msg = response.data;
            let msgType = "error";
            if (msg == "提交成功") msgType = "success";
            Message({
              message: msg,
              type: msgType
            });
          }.bind(this))
          .catch(function (error) {
            console.log(error);
          });
    },
    //全屏
    pushFullScreen() {
      this.map.toggleFullscreen();
    },
    //定位
    pushLocation() {
      const h = this.$createElement;
      this.$msgbox({
        title: "坐标定位",
        message: h('div', {
          attrs: {
            class: "loc-content"
          }
        }, [
          h('div', null, [
            h('form', null, [
              h('div', {
                attrs: {
                  style: "display:block"
                }
              }, [
                h('p', null, [
                  h('span', null, "经度："),
                  h('input', {
                    attrs: {
                      type: "text",
                      id: "yourLon"
                    }
                  }, null)
                ]),
                h('p', null, [
                  h('span', null, "纬度："),
                  h('input', {
                    attrs: {
                      type: "text",
                      id: "yourLat"
                    }
                  }, null)
                ])
              ])
            ])
          ])
        ]),
        showCancelButton: true,
        confirmButtonText: "定位",
        cancelButtonText: "取消"
      }).then(() => {
        var that = this;
        var yourLon = document.getElementById("yourLon").value;
        var yourLat = document.getElementById("yourLat").value;
        console.log(yourLon, yourLat)
        //移动地图中心点
        that.map.setView([yourLat, yourLon], 11);
        //设置中心点图标
        var centerIcon = L.Icon.extend({
          options: {
            iconUrl: require('../assets/loc_red.png'),
            iconSize: [30, 30]
          }
        });
        L.marker([yourLat, yourLon], {
          icon: new centerIcon()
        }).addTo(that.map)
      });
    },
    //计算线段长度
    formatLength(latlng1, latlng2) {
      let lat1 = latlng1.lat;
      let lon1 = latlng1.lng;
      let lat2 = latlng2.lat;
      let lon2 = latlng2.lng;
      let R = 6371;
      let dLat = this.deg2rad(lat2 - lat1);
      let dLon = this.deg2rad(lon2 - lon1);
      let a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(this.deg2rad(lat1)) *
          Math.cos(this.deg2rad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
      var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      var d = R * c;
      return (d / 1.852).toFixed(2);
    },
    //距离中间公式
    deg2rad(deg) {
      return deg * (Math.PI / 180);
    },
    //计算多边形面积
    formatArea(polygon) {
      let seeArea = L.GeometryUtil.geodesicArea(polygon);
      let area = (seeArea / 10e5).toFixed(2) + "km2";
      return area;
    },
    //初始化绘制控件
    initDrawCtrl() {
      //初始化绘制控件
      this.drawControl = new L.Control.Draw({
        position: 'topright',
        draw: {
          polyline: true,
          polygon: true,
          rectangle: false,
          circle: false,
          marker: true,
          circlemarker: false
        }
      }).addTo(this.map);
      L.drawLocal.draw.handlers.polyline = {
        tooltip: {
          start: "点击地图开始画线",
          cont: "继续选择",
          end: "双击完成绘制"
        }
      };
      L.drawLocal.draw.handlers.polygon = {
        tooltip: {
          start: "点击地图开始绘制多边形",
          cont: "继续选择",
          end: "点击第一个顶点完成绘制"
        }
      };
      //创建图层组
      if (!this.drawLayerGroup) {
        this.drawLayerGroup = new L.FeatureGroup();
        this.map.addLayer(this.drawLayerGroup);
      }
      //初始化绘制距离
      var drawDis = 0.0;
      //添加绘制完监听事件
      var that = this;
      this.map.on(L.Draw.Event.CREATED, event => {
        // console.log(event);
        const {layer, layerType} = event;
        if (layerType == 'polygon') {
          let latlng = layer.getLatLngs()[0];
          //计算多边形面积
          let area = this.formatArea(latlng);
          this.addMeasureMarker(area, [latlng[0].lat, latlng[0].lng]);
        }
        drawDis = 0.0;
        this.drawLayerGroup.addLayer(layer);
      });

      //监听选择绘制类型事件
      this.map.on(L.Draw.Event.DRAWSTART, event => {
        const {layerType} = event;
        if (layerType == 'polygon') {
          this.drawType = "polygon";
        } else {
          this.drawType = null;
        }
      });

      //监听编辑事件
      this.map.on(L.Draw.Event.DRAWVERTEX, event => {
        if (that.drawType == null) {
          const {layers} = event;
          let layersArray = layers.getLayers();
          let len = layersArray.length;     //当前图层数组长度
          //若当前只有一个图层（一个顶点），直接画图
          if (len > 1) {
            let latlng1 = layersArray[len - 2].getLatLng();
            let latlng2 = layersArray[len - 1].getLatLng();
            let distance = this.formatLength(latlng1, latlng2);
            drawDis += parseFloat(distance);
            this.addMeasureMarker(drawDis.toFixed(2) + "nm", [latlng2.lat, latlng2.lng]);
          }
        }
      })
    },
    //添加标记
    addMeasureMarker(distance, latlngs) {
      console.log(distance)
      var wordIcon = L.divIcon({
        html: distance,
        className: 'my-div-icon',
        iconSize: 20
      });
      L.marker([latlngs[0], latlngs[1]], {icon: wordIcon}).addTo(this.drawLayerGroup);
    },
    //启动绘制
    startDraw(idx) {
      //先取消绘制对象
      if (this.drawObj) {
        this.drawObj.disable();
      }
      switch (idx) {
        case 0: {   //线
          this.drawObj = new L.Draw.Polyline(this.map, this.drawControl.options.draw.polyline);
          break;
        }
        case 1: {   //矩形
          this.drawObj = new L.Draw.Rectangle(this.map, this.drawControl.options.draw.rectangle);
          break;
        }
        case 2: {    //点
          this.drawObj = new L.Draw.Marker(this.map, this.drawControl.options.draw.marker);
          break;
        }
      }
      //启动
      this.drawObj.enable();
    },
    //销毁绘制控件
    destoryDrawCtr() {
      // this.drawControl = null;
      // this.drawObj = null;
      if (this.drawLayerGroup) {
        this.drawLayerGroup.clearLayers();
      }
      // this.map.off(L.Draw.Event.CREATED);
    },
    //初始化地图
    initMap() {
      //初始化地图对象
      this.map = L.map("map", {
        // center: [40.02404009136253, 116.50641060224784],     //地图中心
        // center: [20.91034, 110.6732],
        // center: [10, 120],
        center: [29.782236666666662, 121.94556166666666],
        zoom: 8,    //缩放比例
        zoomControl: false,     //禁用 + - 按钮
        doubleClickZoom: false,    //禁用双击放大
        attributionControl: false,     //移除右下角Leaflet标识
      });
      //初始化地图图层（墨卡托坐标）
      this.tianditu_1_tile = "http://t4.tianditu.gov.cn/DataServer?T=vec_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
      this.tianditu_1_marker = "http://t4.tianditu.gov.cn/DataServer?T=cva_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
      this.tianditu_2_tile = "http://t4.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
      this.tianditu_2_marker = "http://t4.tianditu.gov.cn/DataServer?T=cia_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
      this.tianditu_3_tile = "http://t4.tianditu.gov.cn/DataServer?T=ter_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
      this.tiandutu_3_marker = "http://t4.tianditu.gov.cn/DataServer?T=cta_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
      //天地图影像图底图（墨卡托坐标）
      this.basicLayer = L.tileLayer(
          this.tianditu_1_tile,
      ).addTo(this.map);

      //天地图影像图标记（墨卡托坐标）
      this.markLayer = L.tileLayer(
          this.tianditu_1_marker,
      ).addTo(this.map);
      // this.map.removeLayer(name);     //移除图层

      //添加鹰眼图
      var osmUrl = "http://map.geoq.cn/ArcGIS/rest/services/ChinaOnlineCommunity/MapServer/tile/{z}/{y}/{x}";     //ArcGis地图
      var osm = new L.tileLayer(osmUrl, {minZoom: 0, maxZoom: 13});
      var miniMap = new MiniMap(osm, {toggleDisplay: true, width: 250, height: 250}).addTo(this.map);

      var that = this;
      //定时调用轨迹数据
      // setInterval(function(){that.getTrajData()}, 3000);
      this.getTrajData();

      //初始化绘制轨迹
      this.drawTrajectory();

      //拖动地图事件
      this.map.on("moveend", function (e) {
        that.zoomLevel = that.map.getZoom();
        //警告消息
        if (that.map.getZoom() < 8) {
          Message.warning({
            message: "当前缩放级别无法显示船舶",
            type: "warning"
          });
        }
        that.getTrajData();
      });

      //地图点击事件
      this.map.on("mousemove", function (e) {
        that.mapLon = e.latlng.lng.toFixed(6);
        that.mapLat = e.latlng.lat.toFixed(6);
      });

      //关闭加载动画
      this.loading = false;
    },
    //读取轨迹数据
    getTrajData() {
      // this.loading = true;     //开启加载动画

      if (this.layerGroup == null) {
        this.layerGroup = L.layerGroup().addTo(this.map);
      }
      this.layerGroup.clearLayers();     //删除既有图层组

      if (this.map.getZoom() < 8) {   //若缩放比例小于8，则不显示轨迹点
        this.loading = false;
        return;
      }

      //获取当前视野范围经纬度
      var leftdown_lng = this.map.getBounds().getSouthWest().lng;     //左下角经度
      var leftdown_lat = this.map.getBounds().getSouthWest().lat;     //左下角纬度
      var rightup_lng = this.map.getBounds().getNorthEast().lng;      //右上角经度
      var rightup_lat = this.map.getBounds().getNorthEast().lat;      //右上角纬度

      //向后端发送请求，读取轨迹数据
      var params = new URLSearchParams();
      params.append("leftdown_lng", leftdown_lng);
      params.append("leftdown_lat", leftdown_lat);
      params.append("rightup_lng", rightup_lng);
      params.append("rightup_lat", rightup_lat);
      getPoints(leftdown_lng, leftdown_lat, rightup_lng, rightup_lat)
          .then(function (response) {
            let that = this;
            let points = response.data;     //全部轨迹数据
            that.shipNum = points.length;      //更新区域船舶数量
            for (let i = 0; i < points.length; i++) {
              let lon = parseFloat(points[i][12]);
              let lat = parseFloat(points[i][11]);
              //初始化图标
              this.trajIcon = L.divIcon({
                className: 'my-div-icon',     //自定义icon css样式
                // iconSize: [12, 12]     //点大小
                html: '<div style="transform: rotate(' + parseFloat(points[i][8]) + 'deg);"><svg  height="30" width="10"><polygon points="5,0 5,10 10,30 0,30 5,10" style="fill:#DAE455;stroke:black;stroke-width:1"/></g></svg></div>'    //设置图标形状和方向
              });
              //绘制轨迹点
              let marker = L.marker([lat, lon], {
                icon: this.trajIcon,
              }).addTo(that.layerGroup).on('click', function (e) {
                let curlatlng = e.latlng;
                for (let j = 0; j < points.length; j++) {
                  if (curlatlng.lng === parseFloat(points[j][12]) && curlatlng.lat === parseFloat(points[j][11])) {
                    that.trajMMSI = points[j][1];
                    that.trajSign = points[j][2];
                    that.trajIMO = points[j][3];
                    that.trajVesselType = points[j][4];
                    that.trajVesselLength = points[j][5];
                    that.trajVesselWidth = points[j][6];
                    that.trajDeepMax = points[j][7];
                    that.trajHeading = points[j][8];
                    that.trajCourse = points[j][9];
                    that.trajSpeed = points[j][10];
                    that.trajLat = points[j][11];
                    that.trajLon = points[j][12];
                    that.trajTarget = points[j][13];
                    that.trajTimeStamp = points[j][14];
                    that.trajVesselName = points[j][15];
                    that.trajCountryName = points[j][16];
                    break;
                  }
                }
                that.contShow = true;     //设置内容框可见
              });
              this.loading = false;     //关闭加载动画
            }
          }.bind(this))
          .catch(function (error) {
            console.log(error);
          });
    },
    //切换地图
    changeMap(command) {
      this.loading = true;      //开启加载动画
      if (command == "tianditu_1") {
        this.basicLayer.setUrl(this.tianditu_1_tile, false);
        this.markLayer.setUrl(this.tianditu_1_marker, false);
      } else if (command == "tianditu_2") {
        this.basicLayer.setUrl(this.tianditu_2_tile, false);
        this.markLayer.setUrl(this.tianditu_2_marker, false);
      } else {
        this.basicLayer.setUrl(this.tianditu_3_tile, false);
        this.markLayer.setUrl(this.tianditu_3_marker, false);
      }
      this.loading = false;       //关闭加载动画
    },
    //绘制轨迹
    drawTrajectory() {
      this.loading = true;        //开启加载动画
      var latlngs = [
        [[17.385044, 78.486671], [16.506174, 80.648015], [17.686816, 83.218482]],
        [[13.082680, 80.270718], [12.971599, 77.594563], [15.828126, 78.037279]]
      ];
      var multiPolyLineOptions = {color: 'red'};
      var multiPolyline = L.polyline(latlngs, multiPolyLineOptions);
      multiPolyline.addTo(this.map);
      this.loading = false;         //关闭加载动画
    },
    //虚拟交通网络
    virtualTransNetwork() {
      this.drawer = false;     //关闭抽屉
      var that = this;
      if (!this.$store.state.isLogin) {
        this.$confirm("请先进行登录！", "警告", {
          confirmButtonText: '去登录',
          cancelButtonText: '取消',
          type: 'warning'
        }).then(() => {
          that.$router.push("/login");
        });
        return;
      }

      const h = this.$createElement;
      this.$msgbox({
        title: "参数设置",
        message: h('div', {
          attrs: {
            class: "loc-content"
          }
        }, [
          h('div', null, [
            h('form', null, [
              h('div', {
                attrs: {
                  style: "display:block"
                }
              }, [
                h('p', null, [
                  h('span', null, "搜索半径参数"),
                  h('input', {
                    attrs: {
                      type: "text",
                      id: "yourLon"
                    }
                  }, null)
                ]),
                h('p', null, [
                  h('span', null, "聚类Eps"),
                  h('input', {
                    attrs: {
                      type: "text",
                      id: "yourLat"
                    }
                  }, null)
                ]),
                h('p', null, [
                  h('span', null, "聚类Minpts"),
                  h('input', {
                    attrs: {
                      type: "text",
                      id: "yourLat"
                    }
                  }, null)
                ])
              ])
            ])
          ])
        ]),
        showCancelButton: true,
        confirmButtonText: "开始",
        cancelButtonText: "取消"
      });

      //绘制虚拟交通网络
      var imageBounds = [[30.00104666666667, 122.36369166666668], [29.782236666666662, 121.94556166666666]]
      // L.imageOverlay(require('../assets/cluster_noback.png'), imageBounds).addTo(this.map);
      L.imageOverlay(require('../assets/net.png'), imageBounds).addTo(this.map);
    }
  },
  mounted() {
    //初始化地图
    this.initMap();
    //初始化绘制控件
    this.initDrawCtrl();
    //登录状态判断
    if (localStorage.getItem("Flag") === 'isLogin') {
      this.$store.state.isLogin = true;
      this.$store.state.userName = localStorage.getItem("userName");
    }
    this.zoomLevel = this.map.getZoom();   //初始化缩放级别
  }
}
</script>

<style lang="scss">

.All-Body {
  background-image: url("../assets/background.jpeg");
}

#left, #right {
  width: 50px;
  height: 70px;
  background-color: rgba(0, 0, 255, 0.8);
  position: absolute;
  top: 520px;
  color: #fff;
  text-align: center;
  line-height: 65px;
  font-size: 65px;
  z-index: 99
}

#left {
  left: 0;
}

#right {
  right: 0;
}

.el-header,
.el-footer {
  background-color: #B3C0D1;
  color: #333;
  text-align: center;
  line-height: 60px;
}

.el-aside {
  background-color: rgba(85, 170, 255, 0.5);
  color: #333;
  text-align: center;
  // line-height: 200px;
}

.el-main {
  background-color: rgba(85, 170, 255, 0.5);
  color: #333;
  text-align: center;
}

// body>.el-container {
// 	margin-bottom: 40px;
// }

.el-container:nth-child(5) .el-aside,
.el-container:nth-child(6) .el-aside {
  line-height: 260px;
}

.el-container:nth-child(7) .el-aside {
  line-height: 320px;
}


* {
  padding: 0;
  margin: 0;
}

.title-span {
  font-size: 25px;
  text-align: left;
  font-weight: bold;
  line-height: calc(7vh);
  overflow: hidden;
}

.func-span {
  text-align: center;
  font-size: 18px;
  line-height: calc(7vh);
  color: #ffffff;
}

.el-menu-item:hover {
  background-color: rgba($color: #ffffff, $alpha: 0.2) !important;
}

.rightsection {
  text-align: right;
  line-height: calc(7vh);
  font-size: 18px;
  color: #ffffff;
}

.btn-click {
  padding: 15px;
}

.btn-click:hover {
  cursor: pointer;
  background-color: rgba($color: #ffffff, $alpha: 0.2);
}

.nav-icon {
  color: #ffffff;
  line-height: calc(8vh);
}

.nav-icon:hover {
  cursor: pointer;
}

#map {
  width: 100%;
  height: calc(93vh);
  z-index: 1;
}

.search_box {
  position: absolute;
  width: 430px;
  line-height: 40px;
  border-radius: 2px;
  font-size: 16px;
  top: 10px;
  left: 20px;
  z-index: 1000;
  background: 0 0;
  background-color: blue;
  overflow: hidden;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
}

.pos_r {
  display: block;
  position: relative;
}

.search_box input {
  border: none;
  background: #fff;
  width: 340px;
  padding-left: 20px;
  padding-right: 10px;
  height: 40px;
  line-height: 40px;
  float: left;
  font-size: 14px;
}

.left_redius {
  border-radius: 2px 0 0 2px;
}

.right_radius {
  border-radius: 0 2px 2px 0;
}

.search_btn {
  display: inline-block;
  width: 60px;
  text-align: center;
  background-image: url("../assets/search_btn.png");
  background-color: #2770d4;
  height: 40px;
  line-height: 30px;
  background-position: center center;
  background-repeat: no-repeat;
  float: right;
}

a {
  color: #333;
  text-decoration: none;
}

a:hover {
  cursor: pointer;
}

.clear_input_btn {
  display: block;
  right: 71px;
  width: 20px;
  height: 20px;
  border-radius: 20px;
  background: #ddd;
  top: 10px;
  // color: #fff;
  line-height: 20px;
  text-align: center;
  font-size: 18px;
  position: absolute;
}

input {
  -webkit-writing-mode: horizontal-tb !important;
  text-rendering: auto;
  color: -internal-light-dark(black, white);
  letter-spacing: normal;
  word-spacing: normal;
  text-transform: none;
  text-indent: 0px;
  text-shadow: none;
  display: inline-block;
  text-align: start;
  appearance: auto;
  -webkit-rtl-ordering: logical;
  cursor: text;
  font: 400 13.3333px arial;
}

.content_box {
  z-index: 5688;
  width: 430px;
  height: 325px;
  top: 60px;
  left: 20px;
  display: block;
  margin: 0;
  padding: 0;
  background-color: #fff;
  -webkit-background-clip: content;
  border-radius: 2px;
  position: relative;
  pointer-events: auto;
  box-shadow: 0 2px 2px rgba(0, 0, 0, .25) !important;
}

.content_title {
  padding: 0;
  border: 0;
  background-color: #2770d4;
  border-radius: 0;
  line-height: 26px;
  height: 40px;
  font-size: 14px;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.content_title span img {
  width: 50px;
  height: 26px;
  margin-top: 7px;
  margin-left: 15px;
  float: left;
}

.content_time_span:hover {
  color: blue;
  cursor: pointer;
  text-decoration: underline;
}

.vessel_name {
  margin-left: 10px;
  margin-top: 8px;
  color: #fff;
  float: left;
  font-size: 16px;
}

.close_btn {
  margin-top: 5px;
  margin-right: 10px;
  float: right;
}

.close_btn_img {
  border: none;
  width: 14px;
  height: 14px;
  display: inline-block;
  vertical-align: middle;
}

.content_title_title {
  margin-left: 18px;
  margin-top: 8px;
  color: #fff;
  float: left;
  font-size: 18px;
}

.content_content {
  height: 325px;
  position: relative;
  overflow: auto;
}

.ship_info {
  background: #fff;
  height: 325px;
}

.ship_info_main {
  overflow: hidden;
  overflow-y: auto;
}

.ship_info_message {
  padding: 0 20px;
}

.ship_info_tabs {
  padding-top: 14px;
}

.ship_info_message table {
  width: 410px;
  table-layout: fixed;
  border-collapse: collapse;
  border-spacing: 0;
  border: 0;
  display: table;
  box-sizing: border-box;
  text-indent: initial;
}

tbody {
  display: table-row-group;
  vertical-align: middle;
  border-color: inherit;
}

.ship_info th {
  font-weight: 400;
  text-align: right;
  color: #333333;
  font-size: 15px;
  line-height: 24px;
  width: 80px;
  padding: 8px;
}

.ship_info_message td {
  white-space: nowrap;
  font-size: 15px;
  padding: 8px;
}

.location_control {
  background-color: rgba(255, 255, 255, 0.9);
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
  margin: 0;
  color: #333;
  font: 11px/1.5 "helvetica, Neue", Arial, Helvetica, sans-serif;
  margin-bottom: 10px;
  margin-right: 10px;
  margin-left: 10px;
  padding: 5px !important;
  border-radius: 3px;
  width: 110px;
  position: absolute;
  bottom: 0;
  z-index: 800;
  pointer-events: auto;
}

.location_control .lon,
.location_control .lat {
  width: 100%;
  font-size: 12px;
  font-family: "microsoft yahei";
  text-align: left;
  user-select: none;
}

.zoom_shipNum {
  position: absolute;
  bottom: 68px;
  left: 10px;
  text-align: center;
  border-radius: 3px;
  padding: 1px 3px;
  font-size: 12px;
  background-color: rgba(255, 255, 255, 0.63);
  color: #333;
  box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.27);
  width: 115px;
  z-index: 800;
}

.advice_box {
  position: absolute;
  top: 245px;
  right: 12px;
  z-index: 800;
}

.advice_box:hover {
  cursor: pointer;
}

.advice_btn {
  display: block;
  width: 30px;
  height: 30px;
  background: #fff;
  border-radius: 3px;
  text-align: center;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
}

.leaflet_msg {
  width: 400px;
  height: 300px;
  background-color: #fff;
  position: relative;
  overflow: auto;
}

.leave_message {
  color: #999
}

.leave_message li {
  margin-top: 20px;
  line-height: 30px;
  list-style: none;
}

.leave_message span {
  display: inline-block;
  width: 85px;
  text-align: right;
  font-size: 14px;
  margin-right: 10px;
}

.leave_message input {
  width: 210px;
  height: 20px;
}

.leave_message textarea {
  width: 275px;
  height: 150px;
}

.leave_message input,
.leave_message textarea {
  background: #f6f6f6;
  font-size: 12px;
  padding: 4px 10px;
  border: 1px solid #ddd;
  border-radius: 2px;
}

.fullscreen_box {
  position: absolute;
  top: 165px;
  right: 12px;
  z-index: 800;
}

.fullscreen_box:hover {
  cursor: pointer;
}

.fullscreen_btn {
  display: block;
  width: 30px;
  height: 30px;
  background-color: #fff;
  border-radius: 3px;
  text-align: center;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
}

.location_box {
  position: absolute;
  top: 205px;
  right: 12px;
  z-index: 800;
}

.location_box:hover {
  cursor: pointer;
}

.location_btn {
  display: block;
  width: 30px;
  height: 30px;
  background-color: #fff;
  border-radius: 3px;
  text-align: center;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
}

.loc-content {
  // height: 120px;
  position: relative;
  overflow: auto;
}

.loc-content p {
  text-align: center;
  font-size: 18px;
  color: #666;
}

.loc-content input {
  width: 190px;
  height: 28px;
  line-height: 28px;
  border: 1px solid #ddd;
  background: #f6f6f6;
  border-radius: 2px;
  margin: 20px 5px 0;
  font-size: 18px;
  text-align: center;
}

.math_box {
  position: absolute;
  top: 125px;
  right: 12px;
  z-index: 800;
}

.math_box:hover {
  cursor: pointer;
}

.math_btn {
  display: block;
  width: 30px;
  height: 30px;
  background-color: #fff;
  border-radius: 3px;
  text-align: center;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
}

.my-div-icon {
  font-size: 15px;
  padding: 20px;
}

.mapSelect_box {
  background-color: rgba(255, 255, 255, 0.63);
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.25);
  position: absolute;
  margin-left: 10px;
  bottom: 100px;
  width: 120px;
  text-align: center;
  font: 11px/1.5 "helvetica, Neue", Arial, Helvetica, sans-serif;
  color: #333;
  border-radius: 3px;
  z-index: 800;
}

.mapSelect_box .img_box {
  width: 120px;
  height: 80px;
}

.mapSelect_box .img_box:hover {
  border: 1px solid red;
  cursor: pointer;
}

.mapSelect_box img {
  width: 120px;
  height: 80px;
}
</style>
<style>
.el-collapse-item__wrap .el-collapse-item__content {
  padding-bottom: 0px;
}
</style>


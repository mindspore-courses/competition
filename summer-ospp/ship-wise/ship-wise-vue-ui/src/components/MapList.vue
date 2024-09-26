<template>
  <div id="map" v-loading="loading" element-loading-text="拼命加载中" element-loading-spinner="el-icon-loading"
       element-loading-background="rgba(0, 0, 0, 0.8)">
    <!-- 搜索框 -->
    <div class="search_box">
      <div>
				<span class="pos_r">
					<input type="text" name id="txtKey" placeholder="搜索船舶mmsi" maxlength="40" class="left_redius"
                 autocomplete="off" @keydown="txtShow = true"/>
				</span>
        <a class="search_btn right_radius" @click="searchShip()"></a>
        <a class="clear_input_btn" style="color: #fff;" v-show="txtShow" @click="clearInputTxt()">x</a>
      </div>
    </div>
    <!-- 内容框 -->

    <!-- 位置坐标 -->
    <div class="location_control">
      <div class="lon">经度：{{ mapLon }}</div>
      <div class="lat">纬度：{{ mapLat }}</div>
    </div>
    <!-- 缩放级别及船舶数量显示 -->
    <div class="zoom_shipNum">
      {{ zoomLevel }}级 - {{ shipNum }}艘
    </div>
    <!-- 取消测距 -->
    <div class="math_box" @click="destoryDrawCtr()">
      <img class="math_btn" src="../assets/delete.png" alt="取消测距"/>
    </div>
    <!-- 提意见 -->
    <div class="advice_box" @click="pushAdvice()">
      <img class="advice_btn" src="../assets/contact.png" alt="提意见"/>
    </div>
    <!-- 全屏 -->
    <div class="fullscreen_box" @click="pushFullScreen()">
      <img class="fullscreen_btn" src="../assets/fullscreen.png" alt="全屏"/>
    </div>
    <!-- 定位 -->
    <div class="location_box" @click="pushLocation()">
      <img class="location_btn" src="../assets/location.png" alt="定位"/>
    </div>
    <!-- 地图选择 -->
    <div class="mapSelect_box">
      <el-collapse value="1" accordion>
        <el-collapse-item title="天地图街道图" name="1">
          <div class="img_box" @click="changeMap('tianditu_1')">
            <img src="../assets/jiedao.png" alt="街道图"/>
          </div>
        </el-collapse-item>
        <el-collapse-item title="天地图影像图" name="2">
          <div class="img_box" @click="changeMap('tianditu_2')">
            <img src="../assets/yingxiang.png" alt="影像图"/>
          </div>
        </el-collapse-item>
        <el-collapse-item title="天地图地形图" name="3">
          <div class="img_box" @click="changeMap('tianditu_3')">
            <img src="../assets/dixing.png" alt="地形图"/>
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>
  </div>
</template>
<script>
import {Message, MessageBox} from 'element-ui'
import MiniMap from 'leaflet-minimap'    // 引入鹰眼图组件
import 'leaflet-minimap/dist/Control.MiniMap.min.css'
import Fullscreen from 'leaflet-fullscreen'     // 引入全屏组件
import 'leaflet-fullscreen/dist/leaflet.fullscreen.css'
import 'leaflet-draw'    // 引入绘图组件
import 'leaflet-draw/dist/leaflet.draw.css'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerIcon1 from 'leaflet/dist/images/marker-icon-yellow.png'
import axios from 'axios'
import {getPoints, searchPoint} from "@/assets/js/api/common-api";

export default {
  data() {
    return {
      map: "",     // 地图
      trajIcon: "",     // 图标
      layerGroup1: null,
      layerGroup2: null,
      layerGroup3: null,
      layerGroup: null,    // 船舶图层组
      pic_name1: null,
      status1: 0,

      // point: ""
      tianditu_1_tile: "",     // 天地图矢量底图
      tianditu_1_marker: "",      // 天地图矢量标记
      tianditu_2_tile: "",     // 天地图影像底图
      tianditu_2_marker: "",      // 天地图影像标记
      tianditu_3_tile: "",     // 天地图地形底图
      tianditu_3_marker: "",      // 天地图地形标记
      basicLayer: "",       // 底图图层
      markerLayer: "",      // 注解图层
      loading: true,        // 加载动画
      drawer: false,        // 抽屉显示
      direction: "ltr",     // 抽屉方向
      trajMMSI: "",      // 船舶MMSI
      trajSign: "",      // 船舶呼号
      trajIMO: "",       // 船舶IMO
      trajVesselType: "", // 船舶类型
      trajVesselLength: "",   // 船舶船长
      trajVesselWidth: "",    // 船舶船宽
      trajDeepMax: "",    // 船舶吃水
      trajHeading: "",   // 船舶船首向
      trajCourse: "",    // 船舶船迹向
      trajSpeed: "",     // 船舶航速
      trajLat: "",       // 船舶纬度
      trajLon: "",       // 船舶经度
      trajTarget: "",    // 船舶目的地
      trajTimeStamp: "", // 船舶更新时间
      trajVesselName: "",     // 船舶船名
      countryPic: "",    // 国家图片
      txtShow: false,    // 搜索框的x是否可见
      contShow: false,   // 内容框是否可见
      mapLon: "",     // 经度
      mapLat: "",     // 纬度
      zoomLevel: "",  // 缩放级别
      shipNum: "",    // 船舶数量
      centerIcon: "",     // 中心点图标
      drawControl: null,    // 画图控件
      drawLayerGroup: null,    // 图形图层组
      drawObj: null,     // 绘制对象
      drawType: null,    // 绘制类型
      imagenumber: 0,
      imageurl: require('../assets/ship1.jpg'),

    }
  },
  created() {
    this.$bus.on('ship_data', this.addMark);
    this.$bus.on('ship_data1', this.clickTr)
  },
  methods: {
    clickTr(b, a) {
      this.map.panTo([a, b]);
    },
    searchShip() {
      const txtKey = document.getElementById("txtKey").value;
      if (this.layerGroup3 != null) {
        this.layerGroup3.clearLayers();
      }
      if (this.pic_name1 == null) {
        Message.warning({
          message: "请先选择遥感图像",
          type: "warning"
        });
        return;
      }
      if (txtKey == null) {
        Message.warning({
          message: "请输入 MMSI",
          type: "warning"
        });
        return;
      }
      searchPoint(this.pic_name1, txtKey, aisData => {
        const data = aisData.data
        let b = [];
        for (let i = 0; i < data.length; i++) {
          b.push([data[i][3], data[i][2]]);
        }
        let polyline = [];
        polyline.push(L.polyline(b, {color: 'blue'}));
        this.layerGroup3 = L.layerGroup(polyline);
        this.map.addLayer(this.layerGroup3);
      })
    },
    //全屏
    pushFullScreen() {
      this.map.toggleFullscreen();
    },
    addMark(geometries, Id, ara, bbox, ais_data, mmsi, ais_time, pic_name) {
      let a = [];
      this.pic_name1 = pic_name;
      a.push([ara[0][1], ara[0][0]]);
      a.push([ara[0][5], ara[0][4]]);
      a.push([ara[0][7], ara[0][6]]);
      a.push([ara[0][3], ara[0][2]]);
      if (geometries.length !== Id.length) {
        return;
      }
      if (this.layerGroup1 != null) {
        this.layerGroup1.clearLayers();
      }
      if (this.layerGroup2 != null) {
        this.layerGroup2.clearLayers();
      }
      if (this.layerGroup3 != null) {
        this.layerGroup3.clearLayers();
      }

      let layers = [];
      let layers1 = [];
      for (let i = 0; i < geometries.length; i++) {
        console.log(geometries[i].lat, geometries[i].lng);
        let redIcon = L.icon({
          iconUrl: markerIcon,
          iconSize: [15, 24],
          iconAnchor: [13, 21]
        });
        let layer = new L.marker([geometries[i].lat, geometries[i].lng], {
          icon: redIcon
        });
        let that = this;
        layer.on("click", function (e) {
          that.$bus.emit("ship_data2", pic_name + '/' + Id[i], bbox[i]);
          this.imageurl = 'http://110.40.157.109:8000/detection/getSubPic/?sub_name=' + pic_name + '/' + Id[i];
          const {lng, lat} = e.latlng;
          let popup = L.popup().setLatLng(e.latlng)
              .setContent('<div style="width:400px;font-size:16px;text-align:center;color:#fff">经纬度:[' +
                  lng + ',' + lat + ']</div>'
                  + '<div style = "width:400px;">'
                  + ' <img style ="width:350px;" src="'
                  + this.imageurl
                  + '"/></div>').openOn(that.map);
        });
        layers.push(layer);
      }
      for (let i = 0; i < ais_data.length; i++) {
        let yellowIcon = L.icon({
          iconUrl: markerIcon1,
          iconSize: [15, 24],
          iconAnchor: [13, 21]
        });
        console.log("ais_data", ais_data[i]["lat"]);

        let layer = new L.marker([ais_data[i]["lat"], ais_data[i]["lng"]], {
          icon: yellowIcon
        });
        let that = this;
        layer.on("click", async function (e) {
          let mmsi1 = mmsi[i]
          let ais_time1 = ais_time[i]

          // const { lat, lng } = e.geometries;
          let lat1 = ais_data[i]["lat"]
          let lng1 = ais_data[i]["lng"]
          //通过一些信息获取黄色ais点的数据
          let {data} = await axios.get('http://110.40.157.109:8000/points/getAisInfo/', {
            params: {
              mmsi: mmsi1,
              date: ais_time1,
              longitude: lng1,
              latitude: lat1,
            }
          })
          let popup = L.popup().setLatLng(e.latlng)
              .setContent(
                  '<div style="width:450px;" >' + '<h1>' + 'AIS信息:' + '</h1>' +
                  '<table border="1"; cellspacing="0";cellpadding="0";  style="width:350px; height:200px;align="center";" >' +
                  '<tr align="center">' + '<th>' + '接收时间' + '</th>' + '<th>' + data['s_time'] + '</th>' + '<th>' + 'MMSI' + '</th>' + '<th>' + mmsi1 + '</th>' + '</tr>' +
                  '<tr align="center">' + '<th>' + '航速' + '</th>' + '<th>' + data['sog'] + '</th>' + '<th>' + '经度' + '</th>' + '<th>' + data['longitude'] + '</th>' + '</tr>' +
                  '<tr align="center">' + '<th>' + '维度' + '</th>' + '<th>' + data['latitude'] + '</th>' + '<th>' + '航向' + '</th>' + '<th>' + data['true_heading'] + '</th>' + '</tr>' +
                  '</table>' + '</div>').openOn(that.map);

        });
        layers.push(layer);
      }
      let layer1 = []
      layer1.push(L.polygon(a, {color: 'red'}))
      this.layerGroup1 = L.layerGroup(layers);
      this.layerGroup2 = L.layerGroup(layer1);
      this.layerGroup3 = L.layerGroup(layers1);
      this.map.addLayer(this.layerGroup1);
      this.map.addLayer(this.layerGroup2);
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
        let that = this;
        let yourLon = document.getElementById("yourLon").value;
        let yourLat = document.getElementById("yourLat").value;
        //移动地图中心点
        that.map.setView([yourLat, yourLon], 11);
        //设置中心点图标
        let centerIcon = L.Icon.extend({
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
      let c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      let d = R * c;
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
      // 初始化绘制控件
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
      let drawDis = 0.0;
      //添加绘制完监听事件
      let that = this;
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
      let wordIcon = L.divIcon({
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
          this.tianditu_2_tile,
      ).addTo(this.map);

      //天地图影像图标记（墨卡托坐标）
      this.markLayer = L.tileLayer(
          this.tianditu_1_marker,
      ).addTo(this.map);
      // this.map.removeLayer(name);     //移除图层

      //添加鹰眼图
      let osmUrl = "http://map.geoq.cn/ArcGIS/rest/services/ChinaOnlineCommunity/MapServer/tile/{z}/{y}/{x}";     //ArcGis地图
      let osm = new L.tileLayer(osmUrl, {minZoom: 0, maxZoom: 13});
      let miniMap = new MiniMap(osm, {toggleDisplay: true, width: 250, height: 250}).addTo(this.map);

      let that = this;
      //定时调用轨迹数据
      // setInterval(function(){that.getTrajData()}, 3000);
      this.getTrajData();

      //初始化绘制轨迹
      this.drawTrajectory();

      //拖动地图事件
      this.map.on("moveend", function (e) {
        that.zoomLevel = that.map.getZoom();
        if (that.map.getZoom() < 8 && that.status1 == 0) {
          Message.warning({
            message: "当前缩放级别无法显示船舶",
            type: "warning"
          });
          that.status1 = 1;
        } else if (that.map.getZoom() >= 8 && that.status1 == 1) {
          that.status1 = 0;
        }
        // that.getTrajData();
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
      let leftdown_lng = this.map.getBounds().getSouthWest().lng;     //左下角经度
      let leftdown_lat = this.map.getBounds().getSouthWest().lat;     //左下角纬度
      let rightup_lng = this.map.getBounds().getNorthEast().lng;      //右上角经度
      let rightup_lat = this.map.getBounds().getNorthEast().lat;      //右上角纬度

      //向后端发送请求，读取轨迹数据
      let params = new URLSearchParams();
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
                  if (curlatlng.lng == parseFloat(points[j][12]) && curlatlng.lat == parseFloat(points[j][11])) {
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
        console.log("hhh");
        this.basicLayer.setUrl(this.tianditu_1_tile, false);
        this.markLayer.setUrl(this.tianditu_1_marker, false);
      } else if (command == "tianditu_2") {
        console.log("hhh");
        this.basicLayer.setUrl(this.tianditu_2_tile, false);
        this.markLayer.setUrl(this.tianditu_2_marker, false);
      } else {
        console.log("hhh");
        this.basicLayer.setUrl(this.tianditu_3_tile, false);
        this.markLayer.setUrl(this.tianditu_3_marker, false);
      }
      this.loading = false;       //关闭加载动画
    },
    //绘制轨迹
    drawTrajectory() {
      this.loading = true;        //开启加载动画
      let latlngs = [
        [[17.385044, 78.486671], [16.506174, 80.648015], [17.686816, 83.218482]],
        [[13.082680, 80.270718], [12.971599, 77.594563], [15.828126, 78.037279]]
      ];
      let multiPolyLineOptions = {color: 'red'};
      let multiPolyline = L.polyline(latlngs, multiPolyLineOptions);
      multiPolyline.addTo(this.map);
      this.loading = false;         //关闭加载动画
    },
    //虚拟交通网络
    virtualTransNetwork() {
      this.drawer = false;     //关闭抽屉
      let that = this;
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
      let imageBounds = [[30.00104666666667, 122.36369166666668], [29.782236666666662, 121.94556166666666]]
      // L.imageOverlay(require('../assets/cluster_noback.png'), imageBounds).addTo(this.map);
      L.imageOverlay(require('../assets/net.png'), imageBounds).addTo(this.map);
    },
    pushAdvice() {
      // TODO: 提交建议
    }
  },
  mounted() {
    //初始化地图
    this.initMap();
    //初始化绘制控件
    this.initDrawCtrl();
    //alert("bangbang")
    //登录状态判断
    if (localStorage.getItem("Flag") == 'isLogin') {
      this.$store.state.isLogin = true;
      this.$store.state.userName = localStorage.getItem("userName");
    }
    this.zoomLevel = this.map.getZoom();   //初始化缩放级别
    //alert(this.zoomLevel);
  }


}
</script>

<style scoped>
::v-deep .leaflet-popup-content-wrapper {
  background: rgba(85, 170, 255, 0.9) !important;
  width: 400px;
}
</style>
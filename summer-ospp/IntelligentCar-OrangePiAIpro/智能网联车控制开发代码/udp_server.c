#include "lwip/nettool/misc.h"
#include "lwip/ip4_addr.h"
#include "lwip/netif.h"
#include "lwip/netifapi.h"
#include "lwip/sockets.h"
#include "cmsis_os2.h"
#include "app_init.h"
#include "soc_osal.h"
#include "wifi_connect.h"
#include "pinctrl.h"
#include "gpio.h"
#include "pwm.h"
#include "osal_debug.h"
#include "tcxo.h"
#include "common_def.h"
#include "hal_gpio.h"
#include "systick.h"
#include "watchdog.h"

#define MOTOR_TASK_STACK_SIZE 0x1000
#define MOTOR_TASK_PRIO (osPriority_t)(17)

#define GPIO_IN1 1
#define GPIO_IN2 6
#define GPIO_IN3 11
#define GPIO_IN4 12

#define GPIO_ENA 9
#define GPIO_ENB 10

#define PWM_ENA_CHANNEL 1
#define PWM_ENB_CHANNEL 2

#define PWM_PIN_MODE 1
#define PWM_GROUP_ID 0
#define PWM2_GROUP_ID 1
#include <math.h>
#define CONFIG_WIFI_SSID "yanzeiPhone"
#define CONFIG_WIFI_PWD "125198125198"
#define CONFIG_SERVER_PORT 8888

#define PWM_TOTAL_PERIOD 200

// 舵机相关定义
#define BSP_SG92R_1 2  // 第一个舵机引脚
#define BSP_SG92R_2 3  // 第二个舵机引脚
#define FREQ_TIME 20000
#define COUNT 10 // 通过计算舵机转到对应角度需要发送10个左右的波形

typedef enum {
    MOTOR_STATE_STOP,
    MOTOR_STATE_FORWARD,
    MOTOR_STATE_BACKWARD,
    MOTOR_STATE_LEFT,
    MOTOR_STATE_RIGHT
} motor_direction_t;

typedef enum {
    MOTOR_SPEED_STOP,
    MOTOR_SPEED_LOW,
    MOTOR_SPEED_MEDIUM,
    MOTOR_SPEED_HIGH
} motor_speed_t;

typedef struct {
    motor_direction_t direction;
    motor_speed_t speed;
} motor_state_t;

volatile motor_state_t motor_state = {MOTOR_STATE_STOP, MOTOR_SPEED_MEDIUM};
volatile uint32_t last_command_time_ms = 0;
static volatile bool motor_enabled = false;   // 新增：电机总开关

// 舵机当前角度（-90到90度）
volatile int current_servo_angle_1 = 0;  // 舵机1角度
volatile int current_servo_angle_2 = 0;  // 舵机2角度

static errcode_t pwm_callback(uint8_t channel) {
    return ERRCODE_SUCC;
}

static void configure_gpio_pins(void) {
    uapi_pin_set_mode(GPIO_IN1, PIN_MODE_0);
    uapi_pin_set_mode(GPIO_IN2, PIN_MODE_0);
    uapi_pin_set_mode(GPIO_IN3, PIN_MODE_0);
    uapi_pin_set_mode(GPIO_IN4, PIN_MODE_0);
    uapi_gpio_set_dir(GPIO_IN1, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_dir(GPIO_IN2, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_dir(GPIO_IN3, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_dir(GPIO_IN4, GPIO_DIRECTION_OUTPUT);

    uapi_pin_set_mode(GPIO_ENA, PWM_PIN_MODE);
    uapi_pin_set_mode(GPIO_ENB, PWM_PIN_MODE);

    uapi_pwm_init();
    uint8_t ch1 = PWM_ENA_CHANNEL;
    uint8_t ch2 = PWM_ENB_CHANNEL;
    uapi_pwm_set_group(PWM_GROUP_ID, &ch1, 1);
    uapi_pwm_set_group(PWM2_GROUP_ID, &ch2, 1);

    uapi_pwm_register_interrupt(PWM_ENA_CHANNEL, pwm_callback);
    uapi_pwm_register_interrupt(PWM_ENB_CHANNEL, pwm_callback);
}

// 舵机初始化
static void S92RInit(void) {
    // 初始化第一个舵机
    uapi_pin_set_mode(BSP_SG92R_1, HAL_PIO_FUNC_GPIO);
    uapi_gpio_set_dir(BSP_SG92R_1, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_val(BSP_SG92R_1, GPIO_LEVEL_LOW);
    
    // 初始化第二个舵机
    uapi_pin_set_mode(BSP_SG92R_2, HAL_PIO_FUNC_GPIO);
    uapi_gpio_set_dir(BSP_SG92R_2, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_val(BSP_SG92R_2, GPIO_LEVEL_LOW);
}

// // 设置舵机角度（-90到90度）
// static void SetServoAngle(int servo_num, int angle) {
//     // 限制角度范围
//     if (angle < -90) angle = -90;
//     if (angle > 90) angle = 90;
    
//     // 将角度转换为脉冲宽度（500-2500μs）
//     // 0度对应1500μs，-90度对应500μs，+90度对应2500μs
//     unsigned int duty = 1500 + (angle * 2000 / 180);
    
//     // 选择要控制的舵机引脚
//     int servo_pin = (servo_num == 1) ? BSP_SG92R_1 : BSP_SG92R_2;
    
//     // 发送COUNT个脉冲来设置舵机位置
//     for (int i = 0; i < COUNT; i++) {
//         uapi_gpio_set_val(servo_pin, GPIO_LEVEL_HIGH);
//         uapi_systick_delay_us(duty);
//         uapi_gpio_set_val(servo_pin, GPIO_LEVEL_LOW);
//         uapi_systick_delay_us(FREQ_TIME - duty);
//     }
    
//     // 更新当前角度
//     if (servo_num == 1) {
//         current_servo_angle_1 = angle;
//     } else {
//         current_servo_angle_2 = angle;
//     }
// }

static void SetServoAngle(int servo_num, int angle) {
    // 限制角度范围
    if (angle < -90) angle = -90;
    if (angle > 90) angle = 90;

    // 逐步过渡到目标角度
    int current_angle = (servo_num == 1) ? current_servo_angle_1 : current_servo_angle_2;
    
    // 计算角度变化的步长
    int angle_step = (angle - current_angle) > 0 ? 1 : -1;
    
    // 将角度转换为脉冲宽度（500-2500μs）
    unsigned int duty = 1500 + (angle * 2000 / 180);
    
    // 选择要控制的舵机引脚
    int servo_pin = (servo_num == 1) ? BSP_SG92R_1 : BSP_SG92R_2;

    // 逐步调整角度
    while (current_angle != angle) {
        // 发送脉冲控制
        uapi_gpio_set_val(servo_pin, GPIO_LEVEL_HIGH);
        uapi_systick_delay_us(duty);
        uapi_gpio_set_val(servo_pin, GPIO_LEVEL_LOW);
        uapi_systick_delay_us(FREQ_TIME - duty);

        // 更新当前角度
        current_angle += angle_step;

        // 计算新的脉冲宽度
        duty = 1500 + (current_angle * 2000 / 180);

        // 小步进，增加时间间隔来平滑过渡
        uapi_systick_delay_us(1000); // 延时1ms，每次步进都慢一点
    }

    // 更新当前角度
    if (servo_num == 1) {
        current_servo_angle_1 = angle;
    } else {
        current_servo_angle_2 = angle;
    }
}

static void set_motor_speed(motor_speed_t speed) {
    float duty_cycle = 0.0f;

    switch (speed) {
        case MOTOR_SPEED_LOW:
            duty_cycle = 0.05f;
            break;
        case MOTOR_SPEED_MEDIUM:
            duty_cycle = 0.45f;
            break;
        case MOTOR_SPEED_HIGH:
            duty_cycle = 1.0f;
            break;
        case MOTOR_SPEED_STOP:
        default:
            duty_cycle = 0.0f;
            break;
    }

    uint32_t high_time = (uint32_t)(PWM_TOTAL_PERIOD * duty_cycle);
    uint32_t low_time  = PWM_TOTAL_PERIOD - high_time;

    pwm_config_t pwm_config = {low_time, high_time, 0, 0, true};

    uapi_pwm_close(PWM_ENA_CHANNEL);
    uapi_pwm_open(PWM_ENA_CHANNEL, &pwm_config);

    uapi_pwm_close(PWM_ENB_CHANNEL);
    uapi_pwm_open(PWM_ENB_CHANNEL, &pwm_config);
    uapi_pwm_start_group(PWM_GROUP_ID);
    uapi_pwm_start_group(PWM2_GROUP_ID);
}

static void motor_forward(motor_speed_t speed) {
    uapi_gpio_set_val(GPIO_IN1, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(GPIO_IN2, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN3, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(GPIO_IN4, GPIO_LEVEL_LOW);
    set_motor_speed(speed);
}

static void motor_backward(motor_speed_t speed) {
    uapi_gpio_set_val(GPIO_IN1, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN2, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(GPIO_IN3, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN4, GPIO_LEVEL_HIGH);
    set_motor_speed(speed);
}

static void motor_left(motor_speed_t speed) {
    uapi_gpio_set_val(GPIO_IN1, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN2, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(GPIO_IN3, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(GPIO_IN4, GPIO_LEVEL_LOW);
    set_motor_speed(speed);
}

static void motor_right(motor_speed_t speed) {
    uapi_gpio_set_val(GPIO_IN1, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(GPIO_IN2, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN3, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN4, GPIO_LEVEL_HIGH);
    set_motor_speed(speed);
}

static void motor_stop(void) {
    uapi_gpio_set_val(GPIO_IN1, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN2, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN3, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(GPIO_IN4, GPIO_LEVEL_LOW);
    set_motor_speed(MOTOR_SPEED_STOP);
}

static void apply_motor_state(void) {
    switch (motor_state.direction) {
        case MOTOR_STATE_FORWARD:
            motor_forward(motor_state.speed);
            break;
        case MOTOR_STATE_BACKWARD:
            motor_backward(motor_state.speed);
            break;
        case MOTOR_STATE_LEFT:
            motor_left(motor_state.speed);
            break;
        case MOTOR_STATE_RIGHT:
            motor_right(motor_state.speed);
            break;
        case MOTOR_STATE_STOP:
        default:
            motor_stop();
            break;
    }
}

static void parse_message(const char *message) {
    // 总开关
    if (strstr(message, "on")) {
        motor_enabled = true;
        osal_printk("Motor enabled\n");
        return;
    } else if (strstr(message, "off")) {
        motor_enabled = false;
        motor_state.direction = MOTOR_STATE_STOP;
        motor_stop();
        osal_printk("Motor disabled\n");
        return;
    }

    // 舵机控制
    if (strstr(message, "servo")) {
        int servo_num = 1; // 默认控制第一个舵机
        int angle;
        
        // 解析舵机编号
        const char *servo_ptr = strstr(message, "servo");
        if (servo_ptr) {
            if (strstr(message, "servo1")) {
                servo_num = 1;
            } else if (strstr(message, "servo2")) {
                servo_num = 2;
            }
        }
        
        // 解析角度值
        const char *angle_ptr = strstr(message, "angle:");
        if (angle_ptr) {
            angle = atoi(angle_ptr + 6); // 跳过"angle:"这6个字符
            SetServoAngle(servo_num, angle);
            osal_printk("Servo%d angle set to: %d\n", servo_num, angle);
            return;
        }
        
        // 舵机左右转动命令
        if (strstr(message, "left")) {
            angle = (servo_num == 1) ? current_servo_angle_1 - 5 : current_servo_angle_2 - 5;
            if (angle < -90) angle = -90;
            SetServoAngle(servo_num, angle);
            osal_printk("Servo%d turned left to: %d\n", servo_num, angle);
        } else if (strstr(message, "right")) {
            angle = (servo_num == 1) ? current_servo_angle_1 + 5 : current_servo_angle_2 + 5;
            if (angle > 90) angle = 90;
            SetServoAngle(servo_num, angle);
            osal_printk("Servo%d turned right to: %d\n", servo_num, angle);
        } else if (strstr(message, "center")) {
            SetServoAngle(servo_num, 0); // 归中
            osal_printk("Servo%d centered\n", servo_num);
        }
        
        return;
    }

    if (!motor_enabled) {
        // 未开启时忽略其他命令
        return;
    }

    motor_direction_t new_direction = motor_state.direction;
    bool direction_changed = false;

    // 方向
    if (strstr(message, "carStatus") && strstr(message, "run")) {
        new_direction = MOTOR_STATE_FORWARD;
        direction_changed = true;
    } else if (strstr(message, "carStatus") && strstr(message, "back")) {
        new_direction = MOTOR_STATE_BACKWARD;
        direction_changed = true;
    } else if (strstr(message, "carStatus") && strstr(message, "left")) {
        new_direction = MOTOR_STATE_LEFT;
        direction_changed = true;
    } else if (strstr(message, "carStatus") && strstr(message, "right")) {
        new_direction = MOTOR_STATE_RIGHT;
        direction_changed = true;
    } else if (strstr(message, "carStatus") && strstr(message, "stop")) {
        new_direction = MOTOR_STATE_STOP;
        direction_changed = true;
    }

    if (direction_changed) {
        if (new_direction == MOTOR_STATE_STOP) {
            motor_state.direction = MOTOR_STATE_STOP;
        } else {
            motor_state.direction = new_direction;
            if (motor_state.speed == MOTOR_SPEED_STOP) {
                motor_state.speed = MOTOR_SPEED_MEDIUM; // 默认中速
            }
        }
        last_command_time_ms = osKernelGetTickCount();
        apply_motor_state();
    }

    // 速度
    if (strstr(message, "carSpeed") && strstr(message, "high")) {
        motor_state.speed = MOTOR_SPEED_HIGH;
        last_command_time_ms = osKernelGetTickCount();
        if (motor_state.direction != MOTOR_STATE_STOP) {
            apply_motor_state();
        }
    } else if (strstr(message, "carSpeed") && strstr(message, "medium")) {
        motor_state.speed = MOTOR_SPEED_MEDIUM;
        last_command_time_ms = osKernelGetTickCount();
        if (motor_state.direction != MOTOR_STATE_STOP) {
            apply_motor_state();
        }
    } else if (strstr(message, "carSpeed") && strstr(message, "low")) {
        motor_state.speed = MOTOR_SPEED_LOW;
        last_command_time_ms = osKernelGetTickCount();
        if (motor_state.direction != MOTOR_STATE_STOP) {
            apply_motor_state();
        }
    }
}

static int udp_server_sample_task(void *param) {
    (void)param;
    int sock_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_length = sizeof(client_addr);
    char recvBuf[512];

    configure_gpio_pins();
    S92RInit(); // 初始化舵机
    wifi_connect(CONFIG_WIFI_SSID, CONFIG_WIFI_PWD);

    if ((sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
        osal_printk("Socket creation failed\n");
        return 0;
    }

    int flags = lwip_fcntl(sock_fd, F_GETFL, 0);
    lwip_fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK);

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(CONFIG_SERVER_PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        osal_printk("Bind failed\n");
        lwip_close(sock_fd);
        return 0;
    }

    while (1) {
        bzero(recvBuf, sizeof(recvBuf));
        ssize_t recv_len = recvfrom(sock_fd, recvBuf, sizeof(recvBuf) - 1, 0,
                                    (struct sockaddr *)&client_addr, &addr_length);
        if (recv_len > 0) {
            recvBuf[recv_len] = '\0';
            osal_printk("Received: %s\n", recvBuf);
            parse_message(recvBuf);
        }
        osal_msleep(10);
    }

    lwip_close(sock_fd);
    uapi_pwm_deinit();
    return 0;
}

static void udp_server_sample_entry(void) {
    osThreadAttr_t attr = {
        .name = "udp_server_sample_task",
        .attr_bits = 0U,
        .cb_mem = NULL,
        .cb_size = 0U,
        .stack_mem = NULL,
        .stack_size = MOTOR_TASK_STACK_SIZE,
        .priority = MOTOR_TASK_PRIO
    };

    osThreadNew((osThreadFunc_t)udp_server_sample_task, NULL, &attr);
}

app_run(udp_server_sample_entry);
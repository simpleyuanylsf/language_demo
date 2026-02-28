#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int32, String
import os
import json
import sys
import pyaudio
import termios
import tty
import select
import vosk
import re
import httpx  # <--- 新增这一行
import noisereduce as nr
import numpy as np

# 尝试导入 OpenAI，如果库没装好，给予友好提示
try:
    from openai import OpenAI
except ImportError:
    print("\n❌ 严重错误: 找不到 'openai' 库")
    print("请运行: pip3 install openai==1.35.0\n")
    sys.exit(1)

# ================= 配置区域 =================

# 笔记本上填 None (自动选择默认麦克风)
# 香橙派上通常填 2
MIC_INDEX = None 

# 【请务必检查这里是否替换为了真实的 Key】
API_KEY = "sk-" 

# 阿里云配置
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# Vosk 模型路径 (相对路径)
MODEL_PATH = "model" 

# ===========================================

class NonBlockingConsole(object):
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False

class VoiceCommander:
    def __init__(self):
        rospy.init_node('voice_llm_node', anonymous=False)
        
        # 1. 先把 client 设为 None，防止后面报 "has no attribute" 错误
        self.client = None
        self.vosk_model = None

        # 2. 初始化 Vosk 模型
        script_dir = os.path.dirname(os.path.realpath(__file__))
        abs_model_path = os.path.join(script_dir, "model")
        
        if not os.path.exists(abs_model_path):
            rospy.logerr(f"❌ 找不到模型文件夹: {abs_model_path}")
            sys.exit(1)
        
        try:
            self.vosk_model = vosk.Model(abs_model_path)
            print(f"✅ Vosk 离线模型加载成功")
        except Exception as e:
            print(f"❌ Vosk 模型加载失败: {e}")
            sys.exit(1)

        # 3. 初始化 ROS 话题
        self.mode_pub = rospy.Publisher('/commander/set_mode', Int32, queue_size=10)
        self.text_pub = rospy.Publisher('/commander/voice_text', String, queue_size=10)

# 4. 初始化大模型 (增加详细报错)
        print("-" * 30)
        print("正在连接阿里云千问大模型...")
        try:
            # 【修复】手动创建一个不带代理配置的客户端
            # 这能完美解决 'unexpected keyword argument proxies' 错误
            custom_http_client = httpx.Client()
            
            self.client = OpenAI(
                api_key=API_KEY, 
                base_url=BASE_URL,
                http_client=custom_http_client # <--- 显式传入客户端
            )
            print(f"✅ 大模型客户端初始化成功！")
        except Exception as e:
            print(f"\n❌❌❌ 大模型连接失败！")
            print(f"错误详情: {e}")
            print("提示: 请检查 API_KEY 是否正确，以及网络是否通畅")
            print("-" * 30)
            # 即使失败，self.client 也是 None，不会报错退出

        # 5. 音频参数
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000 

    def record_audio_manual(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            input_device_index=MIC_INDEX,
                            frames_per_buffer=self.CHUNK)
        except OSError as e:
            print(f"\n❌ 麦克风打开失败: {e}")
            return None

        print("\n" + "="*40)
        print("⌨️  控制说明: [v]按住说话  [f]结束并识别")
        
        frames = []
        is_recording = False

        with NonBlockingConsole() as nbc:
            while not rospy.is_shutdown():
                key = nbc.get_data()
                if key == 'v' or key == 'V':
                    if not is_recording:
                        print("\n🔴 正在录音... (请大声说话)")
                        frames = [] 
                        is_recording = True
                elif key == 'f' or key == 'F':
                    if is_recording:
                        print("\n🛑 停止录音，正在降噪处理...")
                        is_recording = False
                        break 
                elif key == 'q':
                    rospy.signal_shutdown("User quit")
                    return None

                if is_recording:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    sys.stdout.write('.')
                    sys.stdout.flush()

        stream.stop_stream()
        stream.close()
        p.terminate()

        if len(frames) == 0: return None
        
        # === 【新增】软件降噪核心逻辑 ===
        
        # 1. 将字节流转换为 numpy 数组
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # 2. 执行降噪
        # prop_decrease=1.0 表示最大程度减去噪音
        # stationary=True 假设噪音是稳定的（如螺旋桨嗡嗡声）
        try:
            reduced_noise_audio = nr.reduce_noise(
                y=audio_np, 
                sr=self.RATE,
                stationary=True,  # 针对无人机这种持续噪音很有效
                prop_decrease=0.9 # 稍微保留一点点细节，防失真
            )
            
            # 3. 转回字节流传给 Vosk
            return reduced_noise_audio.tobytes()
        except Exception as e:
            print(f"降噪处理出错: {e}, 使用原始音频")
            return audio_data

    def recognize_vosk(self, audio_data):
        rec = vosk.KaldiRecognizer(self.vosk_model, self.RATE)
        rec.AcceptWaveform(audio_data)
        res = json.loads(rec.FinalResult())
        return res.get("text", "").replace(" ", "")

    def analyze_intent(self, text):
        # 【关键修复】这里检查 self.client 是否存在
        if self.client is None:
            print("⚠️ [跳过] 大模型未连接，无法分析意图。")
            return 0

        print(f"🧠 理解指令: {text}")
        system_prompt = "你是一个无人机助手。分析指令并输出数字代码：0-待机(IDLE), 1-起飞(TAKEOFF), 2-任务(MISSION), 3-降落(LAND)。规则：只输出一个数字。"
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                temperature=0.01, max_tokens=5
            )
            raw = response.choices[0].message.content.strip()
            numbers = re.findall(r'\d+', raw)
            return int(numbers[0]) if numbers else 0
        except Exception as e:
            print(f"❌ API 请求失败: {e}")
            return 0

    def run(self):
        while not rospy.is_shutdown():
            raw_audio = self.record_audio_manual()
            if raw_audio:
                text = self.recognize_vosk(raw_audio)
                if text:
                    print(f"📝 识别结果: \"{text}\"")
                    self.text_pub.publish(text)
                    mode = self.analyze_intent(text)
                    self.mode_pub.publish(mode)
                    modes = ["待机", "起飞", "任务", "降落"]
                    mode_name = modes[mode] if 0 <= mode < len(modes) else "未知"
                    print(f"🚀 发送指令: {mode_name}")
                else:
                    print("⚠️ 没听到内容")

if __name__ == '__main__':
    try:
        node = VoiceCommander()
        node.run()
    except rospy.ROSInterruptException:
        pass


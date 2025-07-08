import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import functions as func
import joblib
import google.generativeai as genai
from google.generativeai import types
import threading
from datetime import datetime

# ------------------- 초기 설정 -------------------
st.set_page_config(
    page_title="유압시스템 실시간 이상탐지 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 센서 및 모델 관련 기본 정보 ---
sensor_name_list = ['CE', 'CP', 'EPS1', 'FS1', 'FS2', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'SE', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']
sensor_length_dict = {'CE': 60, 'CP': 60, 'EPS1': 6000, 'FS1': 600, 'FS2': 600, 'PS1': 6000, 'PS2': 6000, 'PS3': 6000, 'PS4': 6000, 'PS5': 6000, 'PS6': 6000, 'SE': 60, 'TS1': 60, 'TS2': 60, 'TS3': 60, 'TS4': 60, 'VS1': 60}
character_list = ['cooler', 'pump', 'accumulator']
last_character_list = ['cooler', 'valve', 'pump', 'accumulator']
SENSOR_GROUPS = {
    "압력 센서 (PS1-PS6)": ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6'], 
    "온도 센서 (TS1-TS4)": ['TS1', 'TS2', 'TS3', 'TS4'],
    "유량 센서": ['FS1', 'FS2'], 
    "진동 센서": ['VS1'], 
    "전력 센서": ['EPS1'], 
    "효율 센서": ['CE', 'CP', 'SE']
}

# --- 데이터 로딩 ---
@st.cache_data
def load_sensor_data():
    data = {}
    for sensor_name in sensor_name_list:
        try:
            data[f"{sensor_name}_df"] = pd.read_csv(rf"C:\Users\kkd02\OneDrive\문서\데이터 분석\최종 프로젝트\data6\{sensor_name}_artificial.csv")
        except FileNotFoundError:
            data[f"{sensor_name}_df"] = pd.DataFrame([0] * 6000)
    return data
sensor_data_frames = load_sensor_data()

# --- 공유 데이터 및 잠금 장치 초기화 ---
if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()

if 'shared_state' not in st.session_state:
    st.session_state.shared_state = {
        "is_running": False, "status_text": "'▶ 공정 시작' 버튼을 눌러주세요.", "history_data": pd.DataFrame(),
        "alarm_message": "", "alarm_has_been_shown": False,
        "cooler_state": "unknown", "valve_state": "unknown", "pump_state": "unknown",
        "accumulator_state": "unknown", "count_text": "대기 중...", "progress_bar_value": 0,
        "current_time": 0, "current_row": 1,
        "sensor_data": {name: [] for name in sensor_name_list},
        "system_stats": {
            "total_processes": 0,
            "normal_processes": 0,
            "abnormal_processes": 0,
            "uptime": datetime.now(),
            "last_maintenance": "2024-12-15",
            "next_maintenance": "2024-12-30"
        }
    }

# --- 시스템 통계는 shared_state에 포함되므로 제거 ---

# ------------------- 백그라운드 실행 함수 -------------------
def run_process_in_background(shared_state, lock):
    history_data_local = {"No.": [], "cooler state": [], "valve state": [], "pump state": [], "accumulator state": []}

    for index in range(10):
        with lock:
            if not shared_state["is_running"]: break
            shared_state["alarm_has_been_shown"] = False
        
        df_list = []
        window_tag = 0
        local_sensor_data = {name: [] for name in sensor_name_list}
        source_data_dict = {name: sensor_data_frames[f'{name}_df'].iloc[index].values for name in sensor_name_list}
        alarm_triggered_in_cycle = False

        for i in range(60):
            with lock:
                if not shared_state["is_running"]: break
            
            for name in sensor_name_list:
                start_idx, end_idx = (sensor_length_dict[name] // 60) * i, (sensor_length_dict[name] // 60) * (i + 1)
                local_sensor_data[name].extend(source_data_dict[name][start_idx:end_idx])

            prob_standard = [func.prob_standard_select('CE', 10, 5, wt)[1] for wt in range(10)]
            prob_standard.pop(); prob_standard.append(59)

            if i in prob_standard:
                live_data_dict = {f'{name}_live_df': pd.DataFrame({name: data}) for name, data in local_sensor_data.items()}
                extracted_df = func.feature_extracted_df(10, 5, window_tag, 1, live_data_dict)
                df_list.append(extracted_df)
                
                with lock:
                    for char in character_list:
                        scaler = joblib.load(rf"C:\Users\kkd02\OneDrive\문서\데이터 분석\최종 프로젝트\모델 정리/{window_tag}_{char}_scaler.pkl")
                        model = joblib.load(rf"C:\Users\kkd02\OneDrive\문서\데이터 분석\최종 프로젝트\모델 정리/{window_tag}_{char}_ee_model.pkl")
                        X_live = scaler.transform(extracted_df.values)
                        shared_state[f'{char}_state'] = 'abnormal' if model.predict(X_live) == -1 else 'normal'
                window_tag += 1

            if i == 59 and df_list:
                concated_df = pd.concat(df_list, axis=1)
                history_data_local['No.'].append(index + 1)
                process_has_abnormal = False
                
                with lock:
                    for char in last_character_list:
                        scaler = joblib.load(rf"C:\Users\kkd02\OneDrive\문서\데이터 분석\최종 프로젝트\모델 정리\{char}_scaler.pkl")
                        model = joblib.load(rf"C:\Users\kkd02\OneDrive\문서\데이터 분석\최종 프로젝트\모델 정리\{char}_model.pkl")
                        X_total = scaler.transform(concated_df.values)
                        prediction = model.predict(X_total)
                        
                        if prediction == -1:
                            state = 'abnormal'
                            state_with_emoji = "🔴 abnormal"
                            alarm_triggered_in_cycle = True
                            process_has_abnormal = True
                        else:
                            state = 'normal'
                            state_with_emoji = "🟢 normal"
                        
                        shared_state[f'{char}_state'] = state
                        history_data_local[f'{char} state'].append(state_with_emoji)
                    
                    # 통계 업데이트
                    shared_state["system_stats"]["total_processes"] += 1
                    if process_has_abnormal:
                        shared_state["system_stats"]["abnormal_processes"] += 1
                    else:
                        shared_state["system_stats"]["normal_processes"] += 1
                    
                    shared_state["history_data"] = pd.DataFrame(history_data_local)
                    if alarm_triggered_in_cycle and not shared_state["alarm_has_been_shown"]:
                        shared_state["alarm_message"] = f"경고: 공정 #{index+1}에서 하나 이상의 부품 이상이 감지되었습니다!"
                        shared_state["alarm_has_been_shown"] = True

            with lock:
                shared_state["count_text"] = f"반복 횟수 : {index+1}회"
                shared_state["progress_bar_value"] = round((100 / 60) * (i + 1))
                shared_state["status_text"] = f'진행률: {round((100 / 60) * (i + 1))}%'
                shared_state["current_time"], shared_state["current_row"] = i + 1, index + 1
                shared_state["sensor_data"] = local_sensor_data
            
            time.sleep(0.1)
        if not shared_state["is_running"]: break

    with lock:
        if shared_state["is_running"]: shared_state["status_text"] = "✅ 공정 완료!"
        else: shared_state["status_text"] = "⏹️ 공정이 중지되었습니다."
        shared_state["is_running"] = False

# --- 버튼 콜백 함수 ---
def start_process_callback():
    with st.session_state.lock:
        st.session_state.shared_state = {
            "is_running": True, "status_text": "공정 시작 중...", "history_data": pd.DataFrame(),
            "alarm_message": "", "alarm_has_been_shown": False,
            "cooler_state": "unknown", "valve_state": "unknown", "pump_state": "unknown",
            "accumulator_state": "unknown", "count_text": "대기 중...", "progress_bar_value": 0,
            "current_time": 0, "current_row": 1,
            "sensor_data": {name: [] for name in sensor_name_list},
            "system_stats": st.session_state.shared_state.get("system_stats", {
                "total_processes": 0,
                "normal_processes": 0,
                "abnormal_processes": 0,
                "uptime": datetime.now(),
                "last_maintenance": "2024-12-15",
                "next_maintenance": "2024-12-30"
            })
        }
    thread = threading.Thread(target=run_process_in_background, args=(st.session_state.shared_state, st.session_state.lock), daemon=True)
    thread.start()

def stop_process_callback():
    with st.session_state.lock:
        st.session_state.shared_state["is_running"] = False

# ------------------- 페이지 초기화 함수 -------------------
def clear_page_state():
    """페이지 전환 시 불필요한 상태 정리"""
    keys_to_remove = []
    for key in st.session_state.keys():
        # 페이지별 임시 상태들 제거 (chat 제외)
        if key.startswith(('temp_', 'page_')) and key != 'chat':
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]

# ------------------- 홈 화면 함수 -------------------
def render_home_page():
    st.title("🏭 유압 시스템 실시간 이상탐지 시스템")
    st.markdown("---")
    
    # 시스템 개요
    st.subheader("📊 시스템 개요")
    col1, col2, col3, col4 = st.columns(4)
    
    with st.session_state.lock:
        system_stats = st.session_state.shared_state["system_stats"]
    
    with col1:
        st.metric(
            label="총 공정 수행 횟수",
            value=system_stats["total_processes"],
            delta=None
        )
    
    with col2:
        st.metric(
            label="정상 공정",
            value=system_stats["normal_processes"],
            delta=None
        )
    
    with col3:
        st.metric(
            label="이상 공정",
            value=system_stats["abnormal_processes"],
            delta=None
        )
    
    with col4:
        uptime_hours = (datetime.now() - system_stats["uptime"]).total_seconds() / 3600
        st.metric(
            label="시스템 가동 시간",
            value=f"{uptime_hours:.1f}시간",
            delta=None
        )
    
    # 현재 시스템 상태
    st.subheader("🔧 현재 시스템 상태")
    with st.session_state.lock:
        shared_copy = st.session_state.shared_state.copy()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 부품 상태 표시
        st.markdown("**부품 상태**")
        component_cols = st.columns(4)
        states = {
            "냉각기": shared_copy["cooler_state"], 
            "밸브": shared_copy["valve_state"], 
            "펌프": shared_copy["pump_state"], 
            "축압기": shared_copy["accumulator_state"]
        }
        
        for (name, state), col in zip(states.items(), component_cols):
            if state == "normal":
                color = "#d4edda"
                icon = "🟢"
            elif state == "abnormal":
                color = "#f8d7da"
                icon = "🔴"
            else:
                color = "#e9ecef"
                icon = "⚪"
            
            col.markdown(f'''
                <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0;">{icon} {name}</h4>
                    <p style="margin: 5px 0 0 0; font-weight: bold;">{state.capitalize()}</p>
                </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        # 공정 상태
        st.markdown("**공정 상태**")
        if shared_copy["is_running"]:
            st.success("🟢 공정 실행 중")
            st.progress(shared_copy["progress_bar_value"])
            st.text(shared_copy["status_text"])
        else:
            st.info("⚪ 공정 대기 중")
            st.text(shared_copy["status_text"])
    
    # 센서 그룹 개요
    st.subheader("📡 센서 그룹 개요")
    sensor_info_cols = st.columns(3)
    
    with sensor_info_cols[0]:
        st.markdown("""
        **압력 센서 (6개)**
        - PS1 ~ PS6
        - 유압 시스템 압력 모니터링
        """)
        
    with sensor_info_cols[1]:
        st.markdown("""
        **온도 센서 (4개)**
        - TS1 ~ TS4
        - 시스템 온도 모니터링
        """)
        
    with sensor_info_cols[2]:
        st.markdown("""
        **기타 센서 (7개)**
        - 유량: FS1, FS2
        - 진동: VS1
        - 전력: EPS1
        - 효율: CE, CP, SE
        """)
    
    # 최근 알림
    st.subheader("🚨 최근 알림")
    if shared_copy["alarm_message"]:
        st.error(shared_copy["alarm_message"], icon="🚨")
    else:
        st.info("현재 활성화된 알림이 없습니다.")

# ------------------- UI 그리기 -------------------
# 페이지 헤더
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 탭 또는 사이드바 네비게이션
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "home"

# 사이드바 네비게이션
with st.sidebar:
    st.title("🏭 메뉴")
    
    if st.button("🏠 홈", use_container_width=True, key="sidebar_home"):
        clear_page_state()
        st.session_state.current_tab = "home"
        st.rerun()
    
    if st.button("📊 실시간 이상탐지", use_container_width=True, key="sidebar_monitoring"):
        clear_page_state()
        st.session_state.current_tab = "monitoring"
        st.rerun()
    
    if st.button("📈 센서 그래프", use_container_width=True, key="sidebar_sensors"):
        clear_page_state()
        st.session_state.current_tab = "sensors"
        st.rerun()
    
    if st.button("🤖 AI 어시스턴트", use_container_width=True, key="sidebar_ai"):
        clear_page_state()
        st.session_state.current_tab = "ai"
        st.rerun()
    
    st.markdown("---")
    st.markdown("**시스템 정보**")
    st.text(f"버전: v1.0.0")
    st.text(f"센서 수: {len(sensor_name_list)}개")
    st.text(f"모니터링 부품: 4개")

# 상단 고정 경고 메시지
with st.session_state.lock:
    alarm_message = st.session_state.shared_state.get("alarm_message", "")
if alarm_message and st.session_state.shared_state.get("alarm_has_been_shown", False):
    st.error(alarm_message, icon="🚨")

# 현재 선택된 탭에 따라 페이지 렌더링
if st.session_state.current_tab == "home":
    render_home_page()

elif st.session_state.current_tab == "monitoring":
    st.title("📊 실시간 이상탐지")
    
    # 페이지 클리어를 위한 컨테이너 사용
    with st.container():
        st.subheader("공정 제어")
        col1, col2, _ = st.columns([1, 1, 8])
        with st.session_state.lock:
            is_running = st.session_state.shared_state["is_running"]
        col1.button("▶ 공정 시작", disabled=is_running, on_click=start_process_callback, key="monitoring_start")
        col2.button("⏹️ 공정 중지", disabled=not is_running, on_click=stop_process_callback, key="monitoring_stop")
        
        st.subheader("공정 진행률")
        with st.session_state.lock:
            shared_copy = st.session_state.shared_state.copy()
        st.text(shared_copy["count_text"])
        st.progress(shared_copy["progress_bar_value"])
        st.text(shared_copy["status_text"])
        
        with st.container():
            c1, c2, c3, c4 = st.columns(4)
            states = {
                "Cooler": shared_copy["cooler_state"], 
                "Valve": shared_copy["valve_state"], 
                "Pump": shared_copy["pump_state"], 
                "Accumulator": shared_copy["accumulator_state"]
            }
            for (name, state), col in zip(states.items(), [c1, c2, c3, c4]):
                color = "#e9ecef" if state == "unknown" else "#d4edda" if state == "normal" else "#f8d7da"
                col.markdown(f'''
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0;">{name} State</h4>
                        <h2 style="margin: 5px 0 0 0;">{state.capitalize()}</h2>
                    </div>
                ''', unsafe_allow_html=True)
        
        st.subheader("공정 이력")
        if shared_copy["alarm_message"]:
            st.error(shared_copy["alarm_message"], icon="🚨")
        if not shared_copy["history_data"].empty:
            st.dataframe(shared_copy["history_data"], hide_index=True)

elif st.session_state.current_tab == "sensors":
    st.title("📈 실시간 센서 데이터 그래프")
    
    # 페이지 클리어를 위한 컨테이너 사용
    with st.container():
        with st.session_state.lock:
            is_running = st.session_state.shared_state["is_running"]
            current_row = st.session_state.shared_state["current_row"]
            current_time = st.session_state.shared_state["current_time"]
            sensor_data_copy = st.session_state.shared_state["sensor_data"].copy()
        
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        title = f"실시간 센서 데이터 (Row: {current_row}, Time: {current_time}s)" if (is_running or current_time > 0) else "실시간 센서 데이터 (대기 중)"
        fig.suptitle(title, fontsize=16)
        
        for ax, (group_name, sensors_in_group) in zip(axs.flat, SENSOR_GROUPS.items()):
            ax.set_title(group_name)
            ax.set_xlabel("Data Points")
            ax.set_ylabel("Value")
            plotted = False
            for sensor in sensors_in_group:
                if sensor_data_copy.get(sensor):
                    ax.plot(sensor_data_copy[sensor], label=sensor)
                    plotted = True
            if plotted:
                ax.legend(loc='upper right')
            ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)

elif st.session_state.current_tab == "ai":
    st.title("🤖 유압 시스템 전문가 AI 어시스턴트")
    
    # 페이지 클리어를 위한 컨테이너 사용
    with st.container():
        try:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=GOOGLE_API_KEY)
        except (FileNotFoundError, KeyError):
            st.warning("Gemini API 키를 찾을 수 없습니다. secrets.toml 파일에 키를 추가하거나 아래에 직접 입력해주세요.")
            GOOGLE_API_KEY = st.text_input("Gemini API 키를 입력하세요:", type="password", key="api_key_input")
            if GOOGLE_API_KEY:
                genai.configure(api_key=GOOGLE_API_KEY)
            else:
                st.info("Gemini API 키를 입력해야 AI 어시스턴트를 사용할 수 있습니다.")
                st.stop()
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="너는 유압 시스템 전문가 AI 어시스턴트야. 사용자가 유압 시스템의 이상 징후, 유지보수, 문제 해결에 대해 질문하면 전문적이고 실용적인 조언을 제공해. 부품: 냉각기(Cooler), 밸브(Valve), 펌프(Pump), 유압(Hydraulic) 센서: 압력(PS), 온도(TS), 유량(FS), 진동(VS), 전력(EPS)"
        )
        
        if "chat" not in st.session_state:
            st.session_state.chat = model.start_chat(history=[])
        
        # 채팅 히스토리 표시
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat.history:
                with st.chat_message("assistant" if msg.role == "model" else msg.role):
                    st.markdown(msg.parts[0].text)
        
        st.subheader("예시 질문")
        questions = [
            "펌프에서 이상이 감지되었습니다. 어떻게 해야 하나요?",
            "냉각기 효율이 떨어졌을 때 점검 사항은?",
            "압력 센서 값이 비정상적으로 높습니다.",
            "예방적 유지보수 일정은 어떻게 수립하나요?"
        ]
        
        cols = st.columns(len(questions))
        for i, q in enumerate(questions):
            if cols[i].button(q, key=f"ex_{i}"):
                st.session_state.chat_prompt = q
                st.rerun()
        
        if prompt := st.chat_input("유압 시스템에 대해 무엇이든 물어보세요..."):
            st.session_state.chat_prompt = prompt
        
        if "chat_prompt" in st.session_state and st.session_state.chat_prompt:
            prompt = st.session_state.chat_prompt
            st.session_state.chat_prompt = None
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                response = st.session_state.chat.send_message(prompt, stream=True)
                st.write_stream(response)
            st.rerun()

# --- 전역 UI 자동 새로고침 루프 ---
with st.session_state.lock:
    is_running_global = st.session_state.shared_state["is_running"]

if is_running_global:
    time.sleep(0.1)
    st.rerun()
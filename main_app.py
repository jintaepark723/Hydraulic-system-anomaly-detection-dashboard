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

# ------------------- 초기 설정 -------------------
st.set_page_config(
    page_title="유압시스템 실시간 이상탐지 시스템",
    layout="wide"
)
st.title(" 유압 시스템 실시간 이상탐지 시스템")

# --- 센서 및 모델 관련 기본 정보 ---
sensor_name_list = ['CE', 'CP', 'EPS1', 'FS1', 'FS2', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'SE', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']
sensor_length_dict = {'CE': 60, 'CP': 60, 'EPS1': 6000, 'FS1': 600, 'FS2': 600, 'PS1': 6000, 'PS2': 6000, 'PS3': 6000, 'PS4': 6000, 'PS5': 6000, 'PS6': 6000, 'SE': 60, 'TS1': 60, 'TS2': 60, 'TS3': 60, 'TS4': 60, 'VS1': 60}
character_list = ['cooler', 'pump', 'accumulator']
last_character_list = ['cooler', 'valve', 'pump', 'accumulator']
SENSOR_GROUPS = {
    "압력 센서 (PS1-PS6)": ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6'], "온도 센서 (TS1-TS4)": ['TS1', 'TS2', 'TS3', 'TS4'],
    "유량 센서": ['FS1', 'FS2'], "진동 센서": ['VS1'], "전력 센서": ['EPS1'], "효율 센서": ['CE', 'CP', 'SE']
}

# --- 데이터 로딩 ---
@st.cache_data
def load_sensor_data():
    data = {}
    for sensor_name in sensor_name_list:
        try:
            data[f"{sensor_name}_df"] = pd.read_csv(rf"test/test_{sensor_name}.csv")
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
        "sensor_data": {name: [] for name in sensor_name_list}
    }

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
                        scaler = joblib.load(rf"models/{window_tag}_{char}_scaler.pkl")
                        model = joblib.load(rf"models/{window_tag}_{char}_ee_model.pkl")
                        X_live = scaler.transform(extracted_df.values)
                        shared_state[f'{char}_state'] = 'abnormal' if model.predict(X_live) == -1 else 'normal'
                window_tag += 1

            if i == 59 and df_list:
                concated_df = pd.concat(df_list, axis=1)
                history_data_local['No.'].append(index + 1)
                with lock:
                    for char in last_character_list:
                        scaler = joblib.load(rf"models\{char}_scaler.pkl")
                        model = joblib.load(rf"models\{char}_oc_model.pkl")
                        X_total = scaler.transform(concated_df.values)
                        prediction = model.predict(X_total)
                        
                        if prediction == -1:
                            state = 'abnormal'
                            state_with_emoji = "🔴 abnormal"
                            alarm_triggered_in_cycle = True
                        else:
                            state = 'normal'
                            state_with_emoji = "🟢 normal"
                        
                        shared_state[f'{char}_state'] = state
                        history_data_local[f'{char} state'].append(state_with_emoji)
                    
                    shared_state["history_data"] = pd.DataFrame(history_data_local)
                    if alarm_triggered_in_cycle and not shared_state["alarm_has_been_shown"]:
                        shared_state["alarm_message"] = f"경고: 공정 #{index+1}에서 하나 이상의 부품 이상이 감지되었습니다!"
                        shared_state["alarm_has_been_shown"] = True

            with lock:
                shared_state["count_text"] = f"반복 횟수 : {index+1}/10회"
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
            "sensor_data": {name: [] for name in sensor_name_list}
        }
    thread = threading.Thread(target=run_process_in_background, args=(st.session_state.shared_state, st.session_state.lock), daemon=True)
    thread.start()

def stop_process_callback():
    with st.session_state.lock:
        st.session_state.shared_state["is_running"] = False

# ------------------- UI 그리기 -------------------
# 상단 고정 경고 메시지
with st.session_state.lock:
    alarm_message = st.session_state.shared_state.get("alarm_message", "")
if alarm_message and st.session_state.shared_state.get("alarm_has_been_shown", False):
    st.error(alarm_message, icon="🚨")

# 탭 생성
tab1, tab2, tab3 = st.tabs([" 실시간 이상탐지", " 실시간 센서 그래프", "AI 어시스턴트"])

with tab1:
    st.subheader("공정 제어")
    col1, col2, _ = st.columns([1, 1, 8])
    with st.session_state.lock:
        is_running = st.session_state.shared_state["is_running"]
    col1.button("▶ 공정 시작", disabled=is_running, on_click=start_process_callback)
    col2.button("⏹️ 공정 중지", disabled=not is_running, on_click=stop_process_callback)
    st.subheader("공정 진행률")
    with st.session_state.lock:
        shared_copy = st.session_state.shared_state.copy()
    st.text(shared_copy["count_text"])
    st.progress(shared_copy["progress_bar_value"])
    st.text(shared_copy["status_text"])
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        states = {"Cooler": shared_copy["cooler_state"], "Valve": shared_copy["valve_state"], "Pump": shared_copy["pump_state"], "Accumulator": shared_copy["accumulator_state"]}
        for (name, state), col in zip(states.items(), [c1, c2, c3, c4]):
            color = "#e9ecef" if state == "unknown" else "#d4edda" if state == "normal" else "#f8d7da"
            col.markdown(f'''<div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;"><h4 style="margin: 0;">{name} State</h4><h2 style="margin: 5px 0 0 0;">{state.capitalize()}</h2></div>''', unsafe_allow_html=True)
    
    st.subheader("공정 이력")
    if shared_copy["alarm_message"]:
        st.error(shared_copy["alarm_message"], icon="🚨")
    if not shared_copy["history_data"].empty:
        st.dataframe(shared_copy["history_data"], hide_index=True)

with tab2:
    st.header("실시간 센서 데이터 그래프")
    with st.session_state.lock:
        is_running = st.session_state.shared_state["is_running"]
        current_row = st.session_state.shared_state["current_row"]
        current_time = st.session_state.shared_state["current_time"]
        sensor_data_copy = st.session_state.shared_state["sensor_data"].copy()
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    title = f"실시간 센서 데이터 (Row: {current_row}, Time: {current_time}s)" if (is_running or current_time > 0) else "실시간 센서 데이터 (대기 중)"
    fig.suptitle(title, fontsize=16)
    for ax, (group_name, sensors_in_group) in zip(axs.flat, SENSOR_GROUPS.items()):
        ax.set_title(group_name); ax.set_xlabel("Data Points"); ax.set_ylabel("Value")
        plotted = False
        for sensor in sensors_in_group:
            if sensor_data_copy.get(sensor):
                ax.plot(sensor_data_copy[sensor], label=sensor); plotted = True
        if plotted: ax.legend(loc='upper right')
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.header("유압 시스템 전문가 AI 어시스턴트")
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    except (FileNotFoundError, KeyError):
        st.warning("Gemini API 키를 찾을 수 없습니다. secrets.toml 파일에 키를 추가하거나 아래에 직접 입력해주세요.")
        GOOGLE_API_KEY = st.text_input("Gemini API 키를 입력하세요:", type="password", key="api_key_input")
        if GOOGLE_API_KEY: genai.configure(api_key=GOOGLE_API_KEY)
        else: st.info("Gemini API 키를 입력해야 AI 어시스턴트를 사용할 수 있습니다."); st.stop()
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction="너는 유압 시스템 전문가 AI 어시스턴트야. 사용자가 유압 시스템의 이상 징후, 유지보수, 문제 해결에 대해 질문하면 전문적이고 실용적인 조언을 제공해. 부품: 냉각기(Cooler), 밸브(Valve), 펌프(Pump), 유압(Hydraulic) 센서: 압력(PS), 온도(TS), 유량(FS), 진동(VS), 전력(EPS)")
    if "chat" not in st.session_state: st.session_state.chat = model.start_chat(history=[])
    for msg in st.session_state.chat.history:
        with st.chat_message("assistant" if msg.role == "model" else msg.role): st.markdown(msg.parts[0].text)
    st.subheader("예시 질문")
    questions = ["펌프에서 이상이 감지되었습니다. 어떻게 해야 하나요?", "냉각기 효율이 떨어졌을 때 점검 사항은?", "압력 센서 값이 비정상적으로 높습니다.", "예방적 유지보수 일정은 어떻게 수립하나요?"]
    cols = st.columns(len(questions))
    for i, q in enumerate(questions):
        if cols[i].button(q, key=f"ex_{i}"): st.session_state.chat_prompt = q; st.rerun()
    if prompt := st.chat_input("유압 시스템에 대해 무엇이든 물어보세요..."): st.session_state.chat_prompt = prompt
    if "chat_prompt" in st.session_state and st.session_state.chat_prompt:
        prompt = st.session_state.chat_prompt
        st.session_state.chat_prompt = None
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.session_state.chat.send_message(prompt, stream=True)
            st.write_stream(response)
        st.rerun()

# --- 전역 UI 자동 새로고침 루프 ---
with st.session_state.lock:
    is_running_global = st.session_state.shared_state["is_running"]

if is_running_global:
    time.sleep(0.2)
    st.rerun()

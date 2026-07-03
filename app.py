import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 웹 페이지 화면 꾸미기 ---
st.set_page_config(page_title="HR AI 분류 시스템", layout="centered")

st.title("🏢 HR 인공지능 일차 분류 시스템")
st.write("인사 데이터 엑셀 파일을 업로드하면 AI가 핵심 인재를 일차적으로 분류합니다.")

# --- 1. 파일 업로드 기능 ---
uploaded_file = st.file_uploader("인사 데이터 엑셀 파일을 선택하세요", type=["xlsx", "csv"])

# 테스트용 샘플 데이터 다운로드 버튼 (파일이 없을 때를 대비)
st.info("💡 테스트해 볼 파일이 없다면? 아래 버튼을 눌러 샘플 데이터를 먼저 다운로드해 보세요.")
sample_data = pd.DataFrame({
    '직원ID': [f'HR_{i:03d}' for i in range(1, 11)],
    '서류_면접_점수': [85, 90, 65, 70, 95, 60, 88, 72, 91, 78],
    '업무_성과_평가점수': [92, 88, 70, 60, 95, 50, 85, 90, 65, 80],
    '프로젝트_참여수': [5, 6, 2, 3, 8, 1, 4, 5, 3, 4],
    '근무_태도_점수': [90, 95, 80, 75, 92, 70, 88, 85, 70, 85]
})
st.download_button("📥 샘플 엑셀 파일 다운로드", sample_data.to_csv(index=False).encode('utf-8'), "sample_hr.csv", "text/csv")


# --- 2. 파일이 업로드되었을 때 작동할 로직 ---
if uploaded_file is not None:
    # 엑셀 혹은 CSV 읽기
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    st.success("✅ 파일 업로드 완료! 데이터를 분석합니다.")
    st.dataframe(df.head()) # 업로드된 데이터 화면에 보여주기
    
    # AI 분류 버튼
    if st.button("🚀 AI 일차 분류 시작하기"):
        
        # 가상의 AI 모델 학습 (패턴 매칭용 단순 기준)
        df['일차_분류_결과'] = np.where((df['업무_성과_평가점수'] >= 85) & (df['근무_태도_점수'] >= 85), 1, 0)
        X = df[['서류_면접_점수', '업무_성과_평가점수', '프로젝트_참여수', '근무_태도_점수']]
        y = df['일차_분류_결과']
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # 예측 및 결과 반영
        df['AI_추천_인재구분'] = model.predict(X)
        df['AI_추천_인재구분'] = df['AI_추천_인재구분'].map({1: '★핵심_관리_대상(고성과)', 0: '일반_유지_대상'})
        
        # 화면에 결과 띄우기
        st.subheader("📊 AI 일차 분류 결과")
        st.dataframe(df[['직원ID', '업무_성과_평가점수', '근무_태도_점수', 'AI_추천_인재구분']])
        
        # 결과 파일 다운로드 버튼 생성
        csv_result = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 분류 결과 엑셀 다운로드",
            data=csv_result,
            file_name="HR_AI_분류_결과.csv",
            mime="text/csv"
        )
import os
import sys

import pandas as pd
from pandas.api.types import CategoricalDtype 

class DataPreprocessor:
    def __init__(self) -> None:
        self._load_data() 

    def _load_data(self):
        # 현재 실행 중인 스크립트 이름 확인
        current_script = os.path.basename(sys.argv[0])  

        # 스크립트 위치에 따라 파일 경로 다르게 설정
        data_path = './data/lgd_raw_230914.csv' if current_script == 'main.py' else '../data/lgd_raw_230914.csv'

        # 데이터 로딩 및 날짜 컬럼 변환
        self.df = pd.read_csv(data_path)
        self.df['CREATION_DATE'] = pd.to_datetime(self.df['CREATION_DATE'])

    def _derive_case_id(self) -> "DataPreprocessor":
        self.df['CaseID'] = self.df.groupby(['SN','ITEM_CODE']).ngroup()
        self.og_df = self.df.copy() # 원본 데이터 저장
        return self
    
    def _simplify_subinventory_label(self) -> "DataPreprocessor":
        self.df['SUBINVENTORY_CODE'] = self.df['SUBINVENTORY_CODE'].str.replace('V-L-','')
        return self
    
    def _set_subinventory_priority_order(self):
        # 우선순위 (원하는 순서로 정렬되게끔)
        priority = ['RMA', 'REP', 'TANA']
        # 우선순위 매핑 딕셔너리 생성
        priority_map = {val: i for i, val in enumerate(priority)}

        # 임시 정렬용 컬럼 생성: 우선순위 값은 낮은 숫자, 나머지는 큰 숫자 부여
        self.df['SUBINVENTORY_ORDER'] = self.df['SUBINVENTORY_CODE'].map(priority_map).fillna(999)

        # 정렬 수행
        self.df = self.df.sort_values(['CaseID', 'CREATION_DATE', 'SUBINVENTORY_ORDER'])

        # 필요 시 정렬용 컬럼 제거
        self.df = self.df.drop(columns='SUBINVENTORY_ORDER')

        return self

    def _sort_by_case_and_time(self) -> "DataPreprocessor":
        self.df = self.df.sort_values(['CaseID','CREATION_DATE','SUBINVENTORY_CODE'])
        return self
    
    def _map_category(self) -> "DataPreprocessor":
        item_map = {
            'LA':'Automobile', 'LB':'Industrial', 'LC':'TV1',  
            'LD':'TV4', 'LE':'TV2', 'LM':'Monitor', 
            'LP':'Laptop', 'LW':'TV3'
                }
        self.df['CATEGORY'] = self.df['ITEM_CODE'].str[:2].map(lambda x: item_map.get(x,'Unkown'))
        return self
    
    def _select_cols(self) -> "DataPreprocessor":
        selected_cols = [
        'CaseID', 'ITEM_CODE', 'SN', 
        'SUBINVENTORY_CODE', 'CATEGORY', 'TXN_TYPE',
        'CREATION_DATE', 'CREATE_BY', 'First',
        'Second', 'Third', 'Fourth'
        ]
        self.df = self.df[selected_cols]
        return self

    def _drop_exact_duplicates(self) -> "DataPreprocessor":
        """
        모든 컬럼 값이 완전히 동일한 중복 행을 제거합니다.
        
        Parameters:
            df (pd.DataFrame): 원본 데이터프레임

        Returns:
            pd.DataFrame: 중복 제거된 데이터프레임
        """
        self.df = self.df.drop_duplicates() 
        return self

    def _remove_conditional_duplicates_with_prev(self) -> "DataPreprocessor": # 테스트 완료
        """
        CaseID, TXN_TYPE, CREATION_DATE 기준 중복(2개) 중:
        - 중복 첫 번째 항목의 바로 이전 행이 존재하고,
        - 이전 행과 중복 첫 항목의 TXN_TYPE, SUBINVENTORY_CODE가 모두 같으면
        → 중복 첫 번째 항목 삭제 (keep='last')
        """
        self.df = self.df.sort_values(['CaseID', 'CREATION_DATE']).reset_index(drop=True)

        dup_key = ['CaseID', 'TXN_TYPE', 'CREATION_DATE']
        dup_df = self.df[self.df.duplicated(dup_key, keep=False)]

        drop_indices = []

        for _, group in dup_df.groupby(dup_key):
            if len(group) != 2:
                continue

            idx1, idx2 = group.index.tolist()

            # 중복 첫 번째 항목의 바로 이전 행이 존재할 때만 처리
            if idx1 > 0:
                prev_idx = idx1 - 1
                is_same_activity = self.df.at[prev_idx, 'TXN_TYPE'] == self.df.at[idx1, 'TXN_TYPE']
                is_same_subinv = self.df.at[prev_idx, 'SUBINVENTORY_CODE'] == self.df.at[idx1, 'SUBINVENTORY_CODE']

                if is_same_activity and is_same_subinv:
                    drop_indices.append(idx1)

        self.df.drop(index=drop_indices, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self
    
    def _remove_time_based_conditional_duplicates(self, time_threshold: int = 5) -> "DataPreprocessor":
        """
        초 단위로 중복 회피한 유사 행 제거 함수 (사용자 지정 시간 허용 범위)

        Parameters:
            time_threshold (int): 중복으로 판단할 시간차 기준 (초 단위, 기본값은 5초)

        Returns:
            self (DataPreprocessor): 정제된 데이터프레임이 반영된 인스턴스
        """

        self.df = self.df.sort_values(['CaseID', 'CREATION_DATE']).reset_index(drop=True)

        # 기준 키 생성
        key_cols = ["CaseID", "TXN_TYPE"]
        self.df["key_base"] = self.df[key_cols].astype(str).agg("_".join, axis=1)

        # 다음 행 기준 비교 값 생성
        self.df["next_time"] = self.df.groupby("key_base")["CREATION_DATE"].shift(-1)
        self.df["next_subinv"] = self.df.groupby("key_base")["SUBINVENTORY_CODE"].shift(-1)

        # 시간 차 계산
        self.df["time_diff"] = (self.df["next_time"] - self.df["CREATION_DATE"]).dt.total_seconds().abs()

        # 제거 대상 인덱스 저장
        drop_indices = []

        for idx in self.df.index:
            if idx + 1 >= len(self.df):
                continue

            current_row = self.df.loc[idx]
            next_row = self.df.loc[idx + 1]

            # 중복 기준 충족 여부
            if (
                current_row["CaseID"] == next_row["CaseID"] and
                current_row["TXN_TYPE"] == next_row["TXN_TYPE"] and
                (next_row["CREATION_DATE"] - current_row["CREATION_DATE"]).total_seconds() <= time_threshold
            ):
                prev_idx = idx - 1
                if prev_idx >= 0:
                    prev_row = self.df.loc[prev_idx]
                    # 이전 행도 같은 조건일 경우 현재 행 제거
                    if (
                        prev_row["CaseID"] == current_row["CaseID"] and
                        prev_row["TXN_TYPE"] == current_row["TXN_TYPE"] and
                        prev_row["SUBINVENTORY_CODE"] == current_row["SUBINVENTORY_CODE"]
                    ):
                        drop_indices.append(idx)

        # 제거 수행
        self.df.drop(index=drop_indices, inplace=True)
        self.df.drop(columns=["key_base", "next_time", "time_diff", "next_subinv"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self

    def _remove_time_near_containerpack_duplicates(self, time_threshold: float = 5.0) -> "DataPreprocessor":
        """ 
        같은 CaseID 내에서 동일 TXN_TYPE + SUBINVENTORY_CODE 조합이 
        time_threshold(초) 이내로 반복되면, 후행 항목을 중복으로 판단하고 제거함.
        
        Parameters:
            time_threshold (float): 시간 차이 임계값 (초 단위)
        """
        df = self.df.copy()
        
        # 키 생성 (중복 판단 기준)
        df["dup_key"] = df[["CaseID", "TXN_TYPE", "SUBINVENTORY_CODE"]].astype(str).agg("_".join, axis=1)
        
        # 정렬 후 이전 시점과 시간 차 계산
        df = df.sort_values(["dup_key", "CREATION_DATE"]).copy()
        df["prev_time"] = df.groupby("dup_key")["CREATION_DATE"].shift()
        df["time_diff"] = (df["CREATION_DATE"] - df["prev_time"]).dt.total_seconds()

        # 중복 판단 (time_threshold 이내)
        dup_idx = df[df["time_diff"] <= time_threshold].index
       
        # 중복 제거
        self.df = df.drop(index=dup_idx).drop(columns=["dup_key", "prev_time", "time_diff"]).reset_index(drop=True)

        return self

    def _finalize(self) -> "DataPreprocessor":
        # self.df = self.df.sort_values(by=['CaseID','CREATION_DATE'])
        self.df = self.df.reset_index(drop=True)
        return self

    def preprocess(self) -> pd.DataFrame:
        obj = (
            self
            ._derive_case_id()
            ._simplify_subinventory_label()
            ._map_category()
            ._sort_by_case_and_time()
            ._select_cols()
            ._drop_exact_duplicates()
            ._remove_conditional_duplicates_with_prev()
            ._remove_time_near_containerpack_duplicates()
            ._remove_time_based_conditional_duplicates()
            ._set_subinventory_priority_order()
            ._finalize()
        )

        return { "processed":obj.df,
                 "original":obj.og_df }

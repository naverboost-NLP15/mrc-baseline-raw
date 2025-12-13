import logging
from typing import List, Dict, Optional
from datasets import load_from_disk, load_dataset, concatenate_datasets, DatasetDict, Dataset
from arguments import DataTrainingArguments

logger = logging.getLogger(__name__)

class DataAssembler:
    """
    학습에 사용할 데이터셋을 로드하고 병합하는 클래스입니다.
    Arguments의 train_datasets 리스트에 정의된 소스들을 읽어와 공통 컬럼 기준으로 병합합니다.
    """
    
    def __init__(self, data_args: DataTrainingArguments):
        self.data_args = data_args
        self.dataset_name = data_args.dataset_name
        self.raw_datasets = {}
        
    def _load_base_dataset(self) -> DatasetDict:
        """기본 데이터셋(competition data)을 로드합니다 (캐싱된 경우 재사용)."""
        logger.info(f"기본 데이터셋 로드: {self.dataset_name}")
        return load_from_disk(self.dataset_name)

    def _get_dataset_source(self, source_name: str, base_datasets: DatasetDict) -> Optional[Dataset]:
        """
        소스 이름에 따라 적절한 데이터셋을 반환합니다.
        
        Args:
            source_name: 'train', 'validation', 'korquad' 등
            base_datasets: load_from_disk로 읽은 기본 데이터셋 객체
        """
        if source_name == "train":
            if "train" in base_datasets:
                return base_datasets["train"]
            else:
                logger.warning("'train' 데이터셋이 기본 데이터셋에 존재하지 않습니다.")
                return None
                
        elif source_name == "validation":
            if "validation" in base_datasets:
                return base_datasets["validation"]
            else:
                logger.warning("'validation' 데이터셋이 기본 데이터셋에 존재하지 않습니다.")
                return None
                
        elif source_name == "korquad":
            logger.info("KorQuad v1 데이터셋 로드 중...")
            korquad = load_dataset("squad_kor_v1")
            return korquad["train"]
            
        else:
            logger.warning(f"알 수 없는 데이터 소스입니다: {source_name}")
            return None

    def get_datasets(self) -> DatasetDict:
        """
        설정된 train_datasets 목록에 따라 데이터를 로드하고 병합하여 반환합니다.
        항상 'train'과 'validation' 키를 가진 DatasetDict를 반환합니다.
        """
        base_datasets = self._load_base_dataset()
        datasets_to_merge = []
        
        # 1. 학습 데이터 병합 준비
        logger.info(f"학습에 사용할 데이터 소스: {self.data_args.train_datasets}")
        
        for source_name in self.data_args.train_datasets:
            dataset = self._get_dataset_source(source_name, base_datasets)
            if dataset is not None:
                datasets_to_merge.append(dataset)
        
        if not datasets_to_merge:
            raise ValueError("학습할 데이터셋이 없습니다. train_datasets 인자를 확인해주세요.")

        # 2. 공통 컬럼 찾기 및 컬럼 선택
        # 모든 데이터셋의 컬럼 교집합을 구합니다.
        common_columns = set(datasets_to_merge[0].column_names)
        for ds in datasets_to_merge[1:]:
            common_columns &= set(ds.column_names)
            
        common_columns = list(common_columns)
        logger.info(f"병합을 위한 공통 컬럼: {common_columns}")
        
        # 3. 데이터셋 병합
        processed_datasets = []
        for ds in datasets_to_merge:
            # 공통 컬럼만 선택
            ds_selected = ds.select_columns(common_columns)
            # Feature 타입 맞추기 (첫 번째 데이터셋 기준)
            if processed_datasets:
                ds_selected = ds_selected.cast(processed_datasets[0].features)
            processed_datasets.append(ds_selected)
            
        combined_train = concatenate_datasets(processed_datasets)
        logger.info(f"최종 학습 데이터 크기: {len(combined_train)}")
        
        # 4. 최종 DatasetDict 구성
        final_datasets = DatasetDict()
        final_datasets["train"] = combined_train
        
        # Validation 셋은 base_datasets의 validation을 그대로 사용 (존재할 경우)
        if "validation" in base_datasets:
            final_datasets["validation"] = base_datasets["validation"]
            
        return final_datasets

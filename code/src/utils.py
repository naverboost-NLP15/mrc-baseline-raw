import json
import os
import traceback
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

MY_SLACK_ID = "U0A1V33UH34"
SLACK_BOT_TOKEN = "xoxb-10047667686263-10059757392309-AGz0QlarAOxod18GrKY5Ion6"

class SlackLogger:
    def __init__(self, user_id=MY_SLACK_ID, token=SLACK_BOT_TOKEN, state_file="latest_thread.json"):
        self.client = WebClient(token=token)
        self.state_file = state_file
        self.user_id = user_id

    def _get_target(self):
        """파일에서 내 ID에 해당하는 스레드 정보를 읽어옴"""
        if not os.path.exists(self.state_file):
            print("❌ 설정 파일이 없습니다.")
            return None, None
        
        with open(self.state_file, "r") as f:
            db = json.load(f)
            
        # 내 ID로 데이터 찾기
        my_data = db.get(self.user_id)
        
        if not my_data:
            print(f"❌ ID '{self.user_id}'에 대한 스레드 정보가 없습니다. 봇을 먼저 멘션해주세요.")
            return None, None
            
        return my_data["channel_id"], my_data["thread_ts"]

    def send(self, message):
        """메시지를 전송하는 함수"""
        channel_id, thread_ts = self._get_target()
        
        if not channel_id:
            return
        
        if isinstance(message, Exception):
            text_payload = f"🚨 에러 발생:\n```{traceback.format_exc()}```"
        else:
            # 일반 메시지는 문자열로 변환
            text_payload = str(message)

        # 빈 문자열 방지 (빈 문자열이면 대체 텍스트 전송)
        if not text_payload.strip():
            text_payload = "(내용 없는 메시지 또는 에러)"

        try:
            self.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=str(message) # 숫자 등이 들어와도 문자열로 변환
            )
        except SlackApiError as e:
            print(f"Slack 전송 실패: {e}")

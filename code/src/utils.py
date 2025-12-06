import os
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import traceback

# 내 ID "U0A1V33UH34"
# 우현님 ID U0A1VMY0REJ

MY_SLACK_ID = "U0A1VMY0REJ"
CHANNEL_ID = "C0A1NPJ2WR1"
SLACK_BOT_TOKEN = "xoxb-10047667686263-10059757392309-AGz0QlarAOxod18GrKY5Ion6"


class SlackLogger:
    def __init__(
        self, token=SLACK_BOT_TOKEN, channel_id=CHANNEL_ID, user_id=MY_SLACK_ID
    ):
        """
        :param token: Bot User OAuth Token
        :param channel_id: 봇을 멘션한 채널 ID (예: C0XXXXXX)
        :param user_id: 내 Slack Member ID (예: U123456)
        """
        self.client = WebClient(token=token)
        self.channel_id = channel_id
        self.user_id = user_id
        self.cached_ts = None  # API 호출 절약을 위한 캐싱

    def _find_my_thread(self):
        """
        슬랙 API를 통해 '내가' 이 채널에서 '봇'을 멘션한 가장 최근 메시지를 찾음
        """
        # 이미 찾은 적이 있다면 재사용 (속도 향상)
        if self.cached_ts:
            return self.cached_ts

        try:
            # 봇 자신의 ID 알아내기
            bot_auth = self.client.auth_test()
            bot_id = bot_auth["user_id"]

            # 채널 내 최근 메시지 50개 조회
            history = self.client.conversations_history(
                channel=self.channel_id, limit=50
            )

            for msg in history.get("messages", []):
                # 조건 1: 작성자가 '나(User)'여야 함
                if msg.get("user") == self.user_id:
                    text = msg.get("text", "")
                    # 조건 2: 내용에 '봇 멘션'이 포함되어야 함
                    if f"<@{bot_id}>" in text:
                        print(f"🔎 스레드 발견! (Time: {msg['ts']})")
                        self.cached_ts = msg["ts"]
                        return msg["ts"]

            print("❌ 최근 50개 메시지 내에서 봇을 멘션한 기록을 찾을 수 없습니다.")
            return None

        except SlackApiError as e:
            print(f"Error finding thread: {e}")
            return None

    def send(self, message):
        # 파일 읽기 대신 API로 찾기
        thread_ts = self._find_my_thread()

        if not thread_ts:
            print("⚠️ 전송 실패: 타겟 스레드를 찾지 못했습니다.")
            return

        # 에러 메시지 처리
        if isinstance(message, Exception):
            text_payload = f"🚨 에러 발생:\n```{traceback.format_exc()}```"
        else:
            text_payload = str(message)

        if not text_payload.strip():
            text_payload = "(내용 없음)"

        try:
            self.client.chat_postMessage(
                channel=self.channel_id, thread_ts=thread_ts, text=text_payload
            )
        except SlackApiError as e:
            print(f"Slack 전송 실패: {e}")

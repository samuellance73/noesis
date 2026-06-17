class MockBot:
    def run(self, token: str):
        print(f"Mock Discord Bot started successfully with token: {token[:4]}... [Mock Mode]")

bot = MockBot()

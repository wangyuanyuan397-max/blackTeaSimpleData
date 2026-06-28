"""
??: src/utils/notification.py
??: ???????
????: ????????????????????
????: ??????????????????
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
import ssl

class Notifier:
    """
    [工具类] 通知发送器。
    
    支持通过 SMTP 发送邮件通知。
    """
    def __init__(self, config):
        """
        初始化通知器。
        
        参数:
            config (dict): 配置字典，包含 email 配置。
                           结构示例:
                           {
                               "smtp_server": "smtp.qq.com",
                               "port": 465,
                               "sender": "xxx@qq.com",
                               "password": "授权码",
                               "receivers": ["yyy@gmail.com"]
                           }
        """
        self.config = config
        self.enabled = config is not None and "smtp_server" in config

    def send_email(self, subject, content, attachment_paths=None):
        """
        发送邮件。
        
        参数:
            subject (str): 邮件标题。
            content (str): 邮件正文。
            attachment_paths (list): 附件文件路径列表。
        """
        if not self.enabled:
            print("[Notifier] Email config not found, skipping notification.")
            return False

        try:
            smtp_server = self.config["smtp_server"]
            port = self.config.get("port", 465)
            sender = self.config["sender"]
            password = self.config["password"]
            receivers = self.config["receivers"]

            # 创建邮件对象
            message = MIMEMultipart()
            # 修复：QQ邮箱等服务器要求 From 标头必须包含 <sender> 格式，或者与登录用户完全一致
            # 简单的 Header(sender, 'utf-8') 可能只会生成 "xxx@qq.com"，
            # 严格模式下建议使用 f"{sender} <{sender}>" 或者直接使用 sender 字符串（smtplib 会处理）
            # 但为了兼容性，最安全的方式是只填邮箱地址，或者 "Nickname <email>"
            
            # 使用 formataddr 构建标准的 From 头: "Sender Name <sender@example.com>"
            from email.utils import formataddr
            message['From'] = formataddr(("AutoScheduler", sender))
            
            message['To'] =  Header(",".join(receivers), 'utf-8')
            message['Subject'] = Header(subject, 'utf-8')

            # 邮件正文
            message.attach(MIMEText(content, 'plain', 'utf-8'))

            # 添加附件
            if attachment_paths:
                for file_path in attachment_paths:
                    if os.path.exists(file_path):
                        att = MIMEText(open(file_path, 'rb').read(), 'base64', 'utf-8')
                        att["Content-Type"] = 'application/octet-stream'
                        # 这里的 filename 可以任意写，写什么名字，邮件中显示什么名字
                        filename = os.path.basename(file_path)
                        att["Content-Disposition"] = f'attachment; filename="{filename}"'
                        message.attach(att)

            # 连接 SMTP 服务器并发送
            if port == 465:
                # SSL 模式
                server = smtplib.SMTP_SSL(smtp_server, port)
            elif port == 587:
                # TLS 模式
                server = smtplib.SMTP(smtp_server, port)
                server.starttls()
            else:
                # 默认非加密
                server = smtplib.SMTP(smtp_server, port)
                
            server.login(sender, password)
            server.sendmail(sender, receivers, message.as_string())
            server.quit()
            
            print(f"[Notifier] Email sent successfully to {receivers}")
            return True
            
        except Exception as e:
            print(f"[Notifier] Failed to send email: {e}")
            import traceback
            traceback.print_exc()
            return False

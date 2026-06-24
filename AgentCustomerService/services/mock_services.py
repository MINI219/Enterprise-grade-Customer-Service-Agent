"""
Mock 业务服务层
提供三个模拟业务函数，返回合理的 JSON 数据，模拟真实客服系统后端
"""
import random
from datetime import datetime, timedelta
from typing import Any, Dict

from app.core.logger import logger


def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    查询订单状态（Mock）

    Args:
        order_id: 订单编号，如 ORD-20240623-001

    Returns:
        包含订单详情、状态、物流进度的模拟数据
    """
    logger.info(f"[MockService] 查询订单状态 | order_id={order_id}")

    # 模拟延迟（可选，注释掉以保持响应迅速）
    # import time; time.sleep(random.uniform(0.1, 0.3))

    statuses = ["pending", "confirmed", "shipped", "out_for_delivery", "delivered"]
    chosen_status = random.choice(statuses)

    progress_map = {
        "pending": [
            {"time": _now_str(), "desc": "订单已提交，等待商户确认"},
        ],
        "confirmed": [
            {"time": _minutes_ago_str(120), "desc": "订单已提交"},
            {"time": _minutes_ago_str(90), "desc": "商户已确认订单"},
        ],
        "shipped": [
            {"time": _minutes_ago_str(180), "desc": "订单已提交"},
            {"time": _minutes_ago_str(150), "desc": "商户已确认订单"},
            {"time": _minutes_ago_str(30), "desc": "包裹已出库，等待揽收"},
        ],
        "out_for_delivery": [
            {"time": _minutes_ago_str(240), "desc": "订单已提交"},
            {"time": _minutes_ago_str(210), "desc": "商户已确认订单"},
            {"time": _minutes_ago_str(120), "desc": "包裹已到达分拣中心"},
            {"time": _minutes_ago_str(40), "desc": "快递员派送中"},
        ],
        "delivered": [
            {"time": _minutes_ago_str(300), "desc": "订单已提交"},
            {"time": _minutes_ago_str(270), "desc": "商户已确认订单"},
            {"time": _minutes_ago_str(180), "desc": "包裹运输中"},
            {"time": _minutes_ago_str(60), "desc": "派送中"},
            {"time": _minutes_ago_str(10), "desc": "已签收"},
        ],
    }

    result = {
        "order_id": order_id,
        "status": chosen_status,
        "product_name": f"智能客服机器人-{random.choice(['标准版', '企业版', '旗舰版'])}",
        "price": round(random.uniform(999.0, 29999.0), 2),
        "created_at": _minutes_ago_str(random.randint(120, 600)),
        "estimated_delivery": _days_later_str(random.randint(1, 7)),
        "progress": progress_map[chosen_status],
    }

    logger.info(f"[MockService] 订单查询成功 | order_id={order_id} | status={chosen_status}")
    return result


def get_logistics_info(tracking_number: str) -> Dict[str, Any]:
    """
    查询物流信息（Mock）

    Args:
        tracking_number: 快递单号，如 SF1234567890

    Returns:
        物流公司、当前位置、完整轨迹
    """
    logger.info(f"[MockService] 查询物流信息 | tracking_number={tracking_number}")

    carriers = ["顺丰速运", "中通快递", "圆通速递", "韵达快递", "京东物流"]
    statuses = ["in_transit", "out_for_delivery", "delivered"]
    chosen_status = random.choice(statuses)

    checkpoint_pool = [
        {"time": _minutes_ago_str(480), "location": "上海分拣中心", "desc": "包裹已发出"},
        {"time": _minutes_ago_str(360), "location": "杭州中转站", "desc": "到达中转站"},
        {"time": _minutes_ago_str(240), "location": "广州分拣中心", "desc": "到达分拣中心"},
        {"time": _minutes_ago_str(120), "location": "深圳配送站", "desc": "到达配送站"},
        {"time": _minutes_ago_str(30), "location": "深圳南山区营业点", "desc": "快递员派送中"},
        {"time": _now_str(), "location": "深圳南山区科技园", "desc": "已签收"},
    ]

    # 按状态截取不同数量的轨迹点
    snipped_map = {
        "in_transit": 3,
        "out_for_delivery": 5,
        "delivered": 6,
    }

    result = {
        "tracking_number": tracking_number,
        "carrier": random.choice(carriers),
        "current_status": chosen_status,
        "origin": random.choice(["上海", "北京", "杭州", "深圳"]),
        "destination": random.choice(["广州", "成都", "武汉", "南京", "深圳"]),
        "checkpoints": checkpoint_pool[: snipped_map[chosen_status]],
    }

    logger.info(
        f"[MockService] 物流查询成功 | tracking_number={tracking_number} | status={chosen_status}"
    )
    return result


def get_complaint_record(user_id: str) -> Dict[str, Any]:
    """
    查询用户投诉记录（Mock）

    Args:
        user_id: 用户 ID，如 U10086

    Returns:
        该用户的投诉列表及处理进度
    """
    logger.info(f"[MockService] 查询投诉记录 | user_id={user_id}")

    subjects = [
        "商品质量问题",
        "物流延迟",
        "客服态度差",
        "退款未到账",
        "商品与描述不符",
        "发票未开具",
    ]
    complaint_statuses = ["submitted", "processing", "resolved", "closed"]

    count = random.randint(0, 3)
    complaints = []
    for i in range(count):
        cs = random.choice(complaint_statuses)
        timeline = [
            {"time": _minutes_ago_str(random.randint(480, 1440)), "desc": "用户提交投诉"},
        ]
        if cs in ("processing", "resolved", "closed"):
            timeline.append(
                {"time": _minutes_ago_str(random.randint(120, 400)), "desc": "客服已受理"}
            )
        if cs in ("resolved", "closed"):
            timeline.append(
                {"time": _minutes_ago_str(random.randint(10, 100)), "desc": "处理完成，已反馈用户"}
            )

        complaints.append(
            {
                "complaint_id": f"CMP-{random.randint(10000, 99999)}",
                "user_id": user_id,
                "subject": random.choice(subjects),
                "status": cs,
                "filed_at": _minutes_ago_str(random.randint(480, 2880)),
                "resolution": (
                            "已与用户沟通，退款 ￥{:.2f} 已原路返回".format(random.uniform(50, 500))
                            if cs == "resolved"
                            else None
                        ),
                "timeline": timeline,
            }
        )

    result = {
        "user_id": user_id,
        "total": count,
        "records": complaints,
    }

    logger.info(f"[MockService] 投诉查询成功 | user_id={user_id} | total={count}")
    return result


# ── 工具函数 ──────────────────────────────────────────────

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _minutes_ago_str(minutes: int) -> str:
    return (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _days_later_str(days: int) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

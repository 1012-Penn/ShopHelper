"""Deterministic demo product catalog shared by catalog-oriented tools."""

from __future__ import annotations

import re

PRODUCT_CATALOG: dict[str, dict] = {
    "SH-E300": {"sku": "SH-E300", "name": "头戴式无线降噪耳机", "price": 299.0, "stock": 120, "status": "在售"},
    "SH-E120": {"sku": "SH-E120", "name": "半入耳无线耳机", "price": 99.0, "stock": 160, "status": "在售"},
    "SH-E200": {"sku": "SH-E200", "name": "入耳式无线耳机", "price": 159.0, "stock": 0, "status": "停产"},
    "SH-K870": {"sku": "SH-K870", "name": "87 键机械键盘", "price": 199.0, "stock": 80, "status": "在售"},
    "SH-K104": {"sku": "SH-K104", "name": "104 键机械键盘", "price": 219.0, "stock": 75, "status": "在售"},
    "SH-K68": {"sku": "SH-K68", "name": "68 键机械键盘", "price": 149.0, "stock": 90, "status": "在售"},
    "SH-B5": {"sku": "SH-B5", "name": "智能手环", "price": 129.0, "stock": 140, "status": "在售"},
    "SH-B7": {"sku": "SH-B7", "name": "智能手环", "price": 179.0, "stock": 110, "status": "在售"},
    "SH-W20": {"sku": "SH-W20", "name": "智能手表", "price": 399.0, "stock": 60, "status": "在售"},
    "SH-W30": {"sku": "SH-W30", "name": "智能手表", "price": 599.0, "stock": 45, "status": "在售"},
    "SH-M10": {"sku": "SH-M10", "name": "无线鼠标", "price": 69.0, "stock": 30, "status": "清仓"},
    "SH-P30": {"sku": "SH-P30", "name": "移动电源", "price": 109.0, "stock": 95, "status": "在售"},
    "SH-C60": {"sku": "SH-C60", "name": "编织数据线", "price": 29.0, "stock": 200, "status": "在售"},
    "SH-S10": {"sku": "SH-S10", "name": "蓝牙音箱", "price": 189.0, "stock": 85, "status": "在售"},
    "SH-T50": {"sku": "SH-T50", "name": "桌面支架", "price": 79.0, "stock": 150, "status": "在售"},
}

_SKU_RE = re.compile(r"SH-[A-Za-z0-9]+", re.IGNORECASE)


def get_product(keyword: str) -> dict:
    raw = str(keyword or "").strip()
    sku_match = _SKU_RE.search(raw.upper())
    if sku_match and sku_match.group(0).upper() in PRODUCT_CATALOG:
        return dict(PRODUCT_CATALOG[sku_match.group(0).upper()])
    compact = raw.casefold()
    matches = [
        row for row in PRODUCT_CATALOG.values()
        if compact and (compact in row["name"].casefold() or row["name"].casefold() in compact)
    ]
    if len(matches) == 1:
        return dict(matches[0])
    return {
        "keyword": raw,
        "name": None,
        "price": None,
        "stock": None,
        "status": "unknown",
        "error": "未找到唯一商品，请提供 SKU",
    }

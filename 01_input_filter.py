import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform


BASE_DIR = Path(__file__).resolve().parent

INPUT_AIS_FILE = BASE_DIR / "ais_6his_6h.csv"
INPUT_ETA_FILE = BASE_DIR / "eta_6his_6h.csv"
CENTERLINE_FILE = BASE_DIR / "downstream_channel_centerline_relaxed_v3_manual_control.geojson"

OUTPUT_FILE = BASE_DIR / "input_filtered.csv"
REPORT_FILE = BASE_DIR / "input_filter_report.csv"

REVERSE_CENTERLINE = False

# 最终点距离中心线过远，认为不在主干航道附近
MAX_FINAL_CROSS_TRACK_KM = 20.0

# 六小时窗口内沿中心线向上海方向的最小推进距离
MIN_DOWNSTREAM_PROGRESS_KM = 0.5

# 连续投影里程增量中，正向增量占比至少达到该阈值
MIN_DOWNSTREAM_POSITIVE_RATIO = 0.55

SHANGHAI_DEST_PREFIXES = {
    'SHANGHA',
    'SH',
    'SHH',
    'SHAIHAI',
    'SHAMGHAI',
    'SHANGGAI',
    'SHANGHAII',
}

SHANGHAI_DEST_EXACT = {
    'SH',
    'SHA',
    'LUODONG',
    'BAOSHAN',
    'BAO',
    'BS',
    'ZHB',
    'JGL',
    'WGQ',
    'WGQ1',
    'WGQ2',
    'WGQ3',
    'WGQ4',
    'WGQ5',
    'W1',
    'W2',
    'W3',
    'W4',
    'W5',
}

SHANGHAI_DEST_CONTAINS = {
    'YANGSHAN',
    'YANG0SHAN',
    'YANGSHANG',
    'WAIGAOQIAO',
    'WGQ',
    'WAIER',
    'WAI2',
    'W2',
    'WAISI',
    'WAI4',
    'W4',
    'WAISIQI',
    'WAIWU',
    'WAI5',
    'W5',
    'WAIYI',
    'WAI1',
    'W1',
    'ZHANGHUABANG',
    'JUNGONGLU',
    'JUNLONGLU',
    'LUODONG',
    'TONGHAIMATOU',
}

SHANGHAI_TERMINAL_CODES = {
    'LD',
    'JGL',
    'WGQ1',
    'WGQ2',
    'WGQ4',
    'WGQ5',
    'YS01',
    'YS03',
    'YS04',
}


def normalize_dest_text(value):
    """归一化目的港文本，用于上海港规则匹配。"""
    if pd.isna(value):
        return ''

    return ''.join(str(value).upper().split())


def is_possible_shanghai_dest(value):
    """根据 AIS/ETA 目的地文本判断是否可能到达上海港。"""
    dest = normalize_dest_text(value)

    if not dest:
        return False

    if any(dest.startswith(prefix) for prefix in SHANGHAI_DEST_PREFIXES):
        return True

    if dest in SHANGHAI_DEST_EXACT:
        return True

    if any(keyword in dest for keyword in SHANGHAI_DEST_CONTAINS):
        return True

    if dest in SHANGHAI_TERMINAL_CODES:
        return True

    return False


def load_centerline(centerline_file, reverse=False):
    """加载中心线并转换到米制局部等距投影。"""
    with open(centerline_file, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    if geojson_data['type'] == 'FeatureCollection':
        line_features = [
            feature
            for feature in geojson_data['features']
            if (
                feature.get('geometry') is not None
                and feature['geometry'].get('type') == 'LineString'
            )
        ]
        if len(line_features) != 1:
            raise ValueError(
                '中心线 GeoJSON 应当包含且只包含一条 LineString，'
                f'实际找到 {len(line_features)} 条'
            )
        centerline_lonlat = shape(line_features[0]['geometry'])
    elif geojson_data['type'] == 'Feature':
        centerline_lonlat = shape(geojson_data['geometry'])
    else:
        centerline_lonlat = shape(geojson_data)

    if centerline_lonlat.geom_type != 'LineString':
        raise ValueError(
            f'中心线必须是 LineString，当前为：{centerline_lonlat.geom_type}'
        )

    if reverse:
        centerline_lonlat = LineString(list(centerline_lonlat.coords)[::-1])

    center = centerline_lonlat.centroid
    local_crs = CRS.from_proj4(
        f'+proj=aeqd '
        f'+lat_0={center.y} '
        f'+lon_0={center.x} '
        f'+datum=WGS84 '
        f'+units=m '
        f'+no_defs'
    )
    transformer = Transformer.from_crs(
        'EPSG:4326',
        local_crs,
        always_xy=True
    )
    centerline_m = transform(transformer.transform, centerline_lonlat)

    return centerline_m, transformer


def project_point_to_centerline(lon, lat, centerline_m, transformer):
    """返回 AIS 点沿中心线里程和横向偏离距离，单位 km。"""
    if pd.isna(lon) or pd.isna(lat):
        return np.nan, np.nan

    x, y = transformer.transform(lon, lat)
    ais_point_m = Point(x, y)
    route_s_m = centerline_m.project(ais_point_m)
    projected_point_m = centerline_m.interpolate(route_s_m)
    cross_track_m = ais_point_m.distance(projected_point_m)

    return route_s_m / 1000.0, cross_track_m / 1000.0


def load_ais_data():
    """读取六小时 AIS 历史，并做基础字段标准化。"""
    df = pd.read_csv(INPUT_AIS_FILE)

    required_cols = [
        'mmsi',
        'postime',
        'lon',
        'lat',
        'sog',
        'draught',
        'cog',
        'dest',
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'{INPUT_AIS_FILE.name} 缺少必要字段：{missing_cols}')

    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce').astype('Int64')
    df['postime'] = pd.to_datetime(df['postime'], utc=True, errors='coerce')

    for col in ['lon', 'lat', 'sog', 'draught', 'cog']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['dest_norm'] = df['dest'].map(normalize_dest_text)
    df['ais_dest_match_shanghai'] = df['dest'].map(is_possible_shanghai_dest)

    df = df.dropna(subset=['mmsi', 'postime', 'lon', 'lat']).copy()
    df = df.sort_values(['mmsi', 'postime']).reset_index(drop=True)

    return df


def load_eta_context():
    """读取 ETA 侧信息，按 mmsi 保留最近一条作为目的地补充。"""
    if not INPUT_ETA_FILE.exists():
        return pd.DataFrame(columns=['mmsi'])

    eta_df = pd.read_csv(INPUT_ETA_FILE)
    if 'mmsi' not in eta_df.columns:
        return pd.DataFrame(columns=['mmsi'])

    eta_df['mmsi'] = pd.to_numeric(eta_df['mmsi'], errors='coerce').astype('Int64')

    time_candidates = ['postime', 'sync_update_time', 'create_time']
    eta_time_col = next(
        (col for col in time_candidates if col in eta_df.columns),
        None
    )
    if eta_time_col is not None:
        eta_df['_eta_context_time'] = pd.to_datetime(
            eta_df[eta_time_col],
            utc=True,
            errors='coerce'
        )
    else:
        eta_df['_eta_context_time'] = pd.NaT

    keep_cols = [
        'mmsi',
        'ais_dest',
        'end_port_code',
        'start_port_code',
        'start_postime',
        'eta',
        'rest_hour',
        'rest_distance',
        '_eta_context_time',
    ]
    keep_cols = [col for col in keep_cols if col in eta_df.columns]

    eta_df = (
        eta_df[keep_cols]
        .dropna(subset=['mmsi'])
        .sort_values(['mmsi', '_eta_context_time'])
        .drop_duplicates(subset=['mmsi'], keep='last')
        .reset_index(drop=True)
    )

    if 'ais_dest' in eta_df.columns:
        eta_df['eta_ais_dest_match_shanghai'] = (
            eta_df['ais_dest'].map(is_possible_shanghai_dest)
        )
    else:
        eta_df['eta_ais_dest_match_shanghai'] = False

    if 'end_port_code' in eta_df.columns:
        eta_df['eta_end_port_match_shanghai'] = (
            eta_df['end_port_code'].map(is_possible_shanghai_dest)
        )
    else:
        eta_df['eta_end_port_match_shanghai'] = False

    return eta_df


def build_vessel_filter_report(ais_df):
    """按船舶生成中心线距离、目的地和下行趋势筛选报告。"""
    centerline_m, transformer = load_centerline(
        CENTERLINE_FILE,
        reverse=REVERSE_CENTERLINE
    )

    projection_result = [
        project_point_to_centerline(lon, lat, centerline_m, transformer)
        for lon, lat in zip(ais_df['lon'], ais_df['lat'])
    ]
    projection_result = pd.DataFrame(
        projection_result,
        columns=['route_s_km', 'cross_track_km'],
        index=ais_df.index
    )
    ais_df = pd.concat([ais_df, projection_result], axis=1)

    ais_df = ais_df.sort_values(['mmsi', 'postime']).reset_index(drop=True)
    ais_df['route_s_delta_km'] = ais_df.groupby('mmsi')['route_s_km'].diff()

    latest_df = (
        ais_df
        .groupby('mmsi', as_index=False)
        .tail(1)
        [[
            'mmsi',
            'postime',
            'lon',
            'lat',
            'dest',
            'dest_norm',
            'ais_dest_match_shanghai',
            'route_s_km',
            'cross_track_km',
        ]]
        .rename(
            columns={
                'postime': 'latest_postime',
                'lon': 'latest_lon',
                'lat': 'latest_lat',
                'dest': 'latest_dest',
                'dest_norm': 'latest_dest_norm',
                'route_s_km': 'last_route_s_km',
                'cross_track_km': 'latest_cross_track_km',
            }
        )
    )

    trend_df = ais_df.groupby('mmsi').agg(
        first_route_s_km=('route_s_km', 'first'),
        last_route_s_km=('route_s_km', 'last'),
        point_count=('route_s_km', 'count'),
        downstream_positive_ratio=(
            'route_s_delta_km',
            lambda x: (x.dropna() > 0).mean() if len(x.dropna()) else 0.0
        ),
    ).reset_index()
    trend_df['route_delta_km'] = (
        trend_df['last_route_s_km'] - trend_df['first_route_s_km']
    )

    report_df = latest_df.merge(trend_df, on='mmsi', how='left')

    eta_context = load_eta_context()
    if not eta_context.empty:
        report_df = report_df.merge(eta_context, on='mmsi', how='left')
    else:
        report_df['eta_ais_dest_match_shanghai'] = False
        report_df['eta_end_port_match_shanghai'] = False

    for col in ['eta_ais_dest_match_shanghai', 'eta_end_port_match_shanghai']:
        if col not in report_df.columns:
            report_df[col] = False
        report_df[col] = report_df[col].fillna(False).astype(bool)

    report_df['destination_match_shanghai'] = (
        report_df['ais_dest_match_shanghai'].fillna(False)
        | report_df['eta_ais_dest_match_shanghai']
        | report_df['eta_end_port_match_shanghai']
    )

    report_df['final_point_near_centerline'] = (
        report_df['latest_cross_track_km'] <= MAX_FINAL_CROSS_TRACK_KM
    )
    report_df['downstream_progress_ok'] = (
        report_df['route_delta_km'] >= MIN_DOWNSTREAM_PROGRESS_KM
    )
    report_df['downstream_ratio_ok'] = (
        report_df['downstream_positive_ratio']
        >= MIN_DOWNSTREAM_POSITIVE_RATIO
    )

    report_df['keep_for_prediction'] = (
        report_df['final_point_near_centerline']
        & report_df['destination_match_shanghai']
        & report_df['downstream_progress_ok']
        & report_df['downstream_ratio_ok']
    )

    report_df['filter_reason'] = 'kept'
    report_df.loc[
        ~report_df['final_point_near_centerline'],
        'filter_reason'
    ] = 'latest_point_far_from_centerline'
    report_df.loc[
        report_df['final_point_near_centerline']
        & ~report_df['destination_match_shanghai'],
        'filter_reason'
    ] = 'destination_not_shanghai'
    report_df.loc[
        report_df['final_point_near_centerline']
        & report_df['destination_match_shanghai']
        & ~report_df['downstream_progress_ok'],
        'filter_reason'
    ] = 'not_enough_downstream_progress'
    report_df.loc[
        report_df['final_point_near_centerline']
        & report_df['destination_match_shanghai']
        & report_df['downstream_progress_ok']
        & ~report_df['downstream_ratio_ok'],
        'filter_reason'
    ] = 'downstream_positive_ratio_too_low'

    return ais_df, report_df


def main():
    ais_df = load_ais_data()
    ais_with_projection_df, report_df = build_vessel_filter_report(ais_df)

    kept_mmsi = set(
        report_df.loc[report_df['keep_for_prediction'], 'mmsi']
    )
    filtered_df = ais_with_projection_df[
        ais_with_projection_df['mmsi'].isin(kept_mmsi)
    ].copy()

    filtered_df.to_csv(OUTPUT_FILE, index=False)
    report_df.to_csv(REPORT_FILE, index=False)

    print(f"原始 AIS 船舶数: {ais_df['mmsi'].nunique()}")
    print(f"保留船舶数: {len(kept_mmsi)}")
    print(f"输出模型输入数据: {OUTPUT_FILE}")
    print(f"输出筛选报告: {REPORT_FILE}")
    print("\n筛选原因分布:")
    print(report_df['filter_reason'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()

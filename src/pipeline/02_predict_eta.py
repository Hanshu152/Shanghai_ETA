import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform


SCRIPT_DIR = Path(__file__).resolve().parent
XGBOOST_DIR = next(
    path for path in [SCRIPT_DIR, *SCRIPT_DIR.parents]
    if path.name == "XGBoost"
)

# PRED_DATA_DIR = XGBOOST_DIR / "data" / "pred_data" / "705"
MODEL_SOURCE_DIR = XGBOOST_DIR / "src" / "model"
PIPELINE_DIR = XGBOOST_DIR / "src" / "pipeline"

# INPUT_FILE = PRED_DATA_DIR / "input_filtered_v1.csv"
VESSEL_FILE = MODEL_SOURCE_DIR / "target_vessels.csv"
CENTERLINE_FILE = MODEL_SOURCE_DIR / "downstream_channel_centerline_relaxed_v3_manual_control.geojson"

MODEL_FILE = PIPELINE_DIR / "xgboost_v8_model.json"
FEATURE_META_FILE = PIPELINE_DIR / "xgboost_v8_features.json"
VESSEL_NAME_FILE = PIPELINE_DIR / "内支线船舶数据表202604140916.xlsx"

# OUTPUT_ALL_FILE = PRED_DATA_DIR / "eta_predictions_all.csv"
DAILY_OUTPUT_TEMPLATE = "{data_date}_eta_for_{date}.csv"

LOCAL_TZ = "Asia/Shanghai"
REVERSE_CENTERLINE = False
GATE_LON = 121.5006300969005
GATE_LAT = 31.425158671076692

XLSX_MAIN_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'

def parse_args():
    parser = argparse.ArgumentParser(
        description="ETA预测流水线"
    )
    parser.add_argument(
        "data_date",
        help = "拉取数据日期，如705"
    )
    return parser.parse_args()
def configure_paths(data_date):
    pred_data_dir = (
        XGBOOST_DIR / "data" / "pred_data" / data_date
    )
    return{
         "pred_data_dir": pred_data_dir,
        "input_ais_file": pred_data_dir / "ais_6his_6h.csv",
        "input_eta_file": pred_data_dir / "eta_6his_6h.csv",
        "input_filtered_file": pred_data_dir / "input_filtered.csv",
        "filter_report_file": pred_data_dir / "input_filter_report.csv",
        "prediction_all_file": pred_data_dir / "eta_predictions_all.csv",
    }

def load_vessel_name_map(vessel_name_file):
    """从 XLSX 首个工作表读取 MMSI 与船名映射，不依赖 openpyxl。"""
    if not vessel_name_file.exists():
        raise FileNotFoundError(f'未找到船名映射文件: {vessel_name_file}')

    ns = {'main': XLSX_MAIN_NS}
    with ZipFile(vessel_name_file) as workbook:
        shared_strings = []
        if 'xl/sharedStrings.xml' in workbook.namelist():
            shared_root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
            for item in shared_root.findall('main:si', ns):
                shared_strings.append(
                    ''.join(
                        node.text or ''
                        for node in item.iterfind('.//main:t', ns)
                    )
                )

        sheet_files = sorted(
            name
            for name in workbook.namelist()
            if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name)
        )
        if not sheet_files:
            raise ValueError(f'工作簿中没有工作表: {vessel_name_file}')

        sheet_root = ET.fromstring(workbook.read(sheet_files[0]))
        rows = []
        for row in sheet_root.findall('.//main:sheetData/main:row', ns):
            row_values = {}
            for cell in row.findall('main:c', ns):
                reference = cell.get('r', '')
                column_match = re.match(r'[A-Z]+', reference)
                if column_match is None:
                    continue

                cell_type = cell.get('t')
                value_node = cell.find('main:v', ns)
                value = '' if value_node is None else value_node.text or ''
                if cell_type == 's' and value:
                    value = shared_strings[int(value)]
                elif cell_type == 'inlineStr':
                    value = ''.join(
                        node.text or ''
                        for node in cell.iterfind('.//main:t', ns)
                    )
                row_values[column_match.group()] = value
            rows.append(row_values)

    if not rows:
        raise ValueError(f'船名映射工作表为空: {vessel_name_file}')

    header = {
        str(value).strip(): column
        for column, value in rows[0].items()
    }
    mmsi_column = header.get('MMSI') or header.get('mmsi')
    vessel_name_column = (
        header.get('船名')
        or header.get('船舶名称')
        or header.get('vessel_name')
    )
    if mmsi_column is None or vessel_name_column is None:
        raise ValueError(
            f'{vessel_name_file.name} 缺少 MMSI 或船名字段'
        )

    vessel_names = pd.DataFrame(
        {
            'mmsi': [row.get(mmsi_column, '') for row in rows[1:]],
            'vessel_name': [
                row.get(vessel_name_column, '')
                for row in rows[1:]
            ],
        }
    )
    vessel_names['mmsi'] = pd.to_numeric(
        vessel_names['mmsi'], errors='coerce'
    ).astype('Int64')
    vessel_names['vessel_name'] = (
        vessel_names['vessel_name'].astype(str).str.strip()
    )
    vessel_names = (
        vessel_names
        .dropna(subset=['mmsi'])
        .loc[lambda data: data['vessel_name'] != '']
        .drop_duplicates(subset=['mmsi'], keep='last')
    )
    return vessel_names.set_index('mmsi')['vessel_name']


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """向量化计算两点间球面距离，单位 km。"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


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


def load_vessel_static(ship_type_features):
    """读取船舶静态特征，并对齐训练阶段保存的船舶类型 one-hot 列。"""
    vessel_static = pd.read_csv(VESSEL_FILE, encoding='utf-8-sig')
    vessel_static['mmsi'] = pd.to_numeric(
        vessel_static['mmsi'],
        errors='coerce'
    ).astype('Int64')

    vessel_static = vessel_static[
        ['mmsi', '船舶类型', '船长', '船宽']
    ].drop_duplicates(subset=['mmsi'])
    vessel_static = vessel_static.rename(
        columns={
            '船舶类型': 'ship_type',
            '船长': 'ship_length',
            '船宽': 'ship_width',
        }
    )

    vessel_static['ship_length'] = pd.to_numeric(
        vessel_static['ship_length'],
        errors='coerce'
    )
    vessel_static['ship_width'] = pd.to_numeric(
        vessel_static['ship_width'],
        errors='coerce'
    )
    vessel_static['ship_type'] = vessel_static['ship_type'].fillna('未知')

    length_median = vessel_static['ship_length'].median()
    width_median = vessel_static['ship_width'].median()
    vessel_static['ship_length'] = vessel_static['ship_length'].fillna(length_median)
    vessel_static['ship_width'] = vessel_static['ship_width'].fillna(width_median)

    ship_type_dummies = pd.get_dummies(
        vessel_static['ship_type'],
        prefix='ship_type',
        dtype=int
    )
    vessel_static = pd.concat(
        [
            vessel_static[['mmsi', 'ship_length', 'ship_width']],
            ship_type_dummies,
        ],
        axis=1
    )

    for col in ship_type_features:
        if col not in vessel_static.columns:
            vessel_static[col] = 0

    keep_cols = ['mmsi', 'ship_length', 'ship_width'] + ship_type_features
    return vessel_static[keep_cols], length_median, width_median


def add_time_rolling_sog(df):
    """按船舶六小时历史计算历史均速和 2h/4h 时间窗口均速。"""
    df = df.sort_values(['voyage_id', 'postime']).copy()
    df['avg_sog_all'] = (
        df.groupby('voyage_id')['sog']
          .expanding()
          .mean()
          .reset_index(level=0, drop=True)
    )

    rolling_parts = []
    for _, group in df.groupby('voyage_id', sort=False):
        group = group.sort_values('postime').copy()
        group_indexed = group.set_index('postime')
        group['avg_sog_2h'] = (
            group_indexed['sog']
            .rolling('2h', min_periods=1)
            .mean()
            .to_numpy()
        )
        group['avg_sog_4h'] = (
            group_indexed['sog']
            .rolling('4h', min_periods=1)
            .mean()
            .to_numpy()
        )
        rolling_parts.append(group)

    return pd.concat(rolling_parts, axis=0).sort_index()


def load_filtered_input():
    """读取 01_input_filter.py 输出的筛选后 AIS 历史。"""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f'未找到 {INPUT_FILE}，请先运行 01_input_filter.py'
        )

    df = pd.read_csv(INPUT_FILE)

    required_cols = ['mmsi', 'postime', 'lon', 'lat', 'sog', 'draught', 'cog']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'{INPUT_FILE.name} 缺少必要字段：{missing_cols}')

    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce').astype('Int64')
    df['voyage_id'] = df['mmsi'].astype(str)
    df['postime'] = pd.to_datetime(df['postime'], utc=True, errors='coerce')

    for col in ['lon', 'lat', 'sog', 'draught', 'cog']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['mmsi', 'postime', 'lon', 'lat']).copy()
    df = df.sort_values(['voyage_id', 'postime']).reset_index(drop=True)

    for col in ['sog', 'draught', 'cog']:
        df[col] = (
            df.groupby('voyage_id')[col]
              .transform(lambda s: s.ffill().bfill())
        )
        fallback = df[col].median()
        if pd.isna(fallback):
            fallback = 0.0
        df[col] = df[col].fillna(fallback)

    return df


def build_model_features(df, features, ship_type_features):
    """把筛选后的 AIS 历史构造成 xgboost_v8 需要的特征。"""
    centerline_m, centerline_transformer = load_centerline(
        CENTERLINE_FILE,
        reverse=REVERSE_CENTERLINE
    )

    vessel_static, length_median, width_median = load_vessel_static(
        ship_type_features
    )
    df = df.merge(vessel_static, on='mmsi', how='left')
    df['ship_length'] = df['ship_length'].fillna(length_median)
    df['ship_width'] = df['ship_width'].fillna(width_median)
    df[ship_type_features] = df[ship_type_features].fillna(0).astype(int)

    df = df.sort_values(['voyage_id', 'postime']).reset_index(drop=True)

    voyage_context = df.groupby('voyage_id').agg(
        start_lon=('lon', 'first'),
        start_lat=('lat', 'first'),
    ).reset_index()
    df = df.merge(voyage_context, on='voyage_id', how='left')

    df['hour_of_day'] = df['postime'].dt.hour
    df['month'] = df['postime'].dt.month

    df['prev_lon'] = df.groupby('voyage_id')['lon'].shift(1).fillna(df['lon'])
    df['prev_lat'] = df.groupby('voyage_id')['lat'].shift(1).fillna(df['lat'])
    df['step_dist_km'] = haversine_vectorized(
        df['prev_lon'],
        df['prev_lat'],
        df['lon'],
        df['lat']
    )
    df['dist_traveled_km'] = df.groupby('voyage_id')['step_dist_km'].cumsum()

    df['prev_postime'] = df.groupby('voyage_id')['postime'].shift(1)
    df['dt_h'] = (
        df['postime'] - df['prev_postime']
    ).dt.total_seconds() / 3600.0
    df['dt_h'] = df['dt_h'].fillna(0).clip(lower=0)
    df['active_step_h'] = np.where(
        df['sog'].between(2.0, 12.0)
        & df['dt_h'].between(0.0, 0.5, inclusive='both'),
        df['dt_h'],
        0.0
    )
    df['pure_hours_elapsed'] = df.groupby('voyage_id')['active_step_h'].cumsum()

    df['port_distance_to_gate'] = haversine_vectorized(
        df['start_lon'],
        df['start_lat'],
        GATE_LON,
        GATE_LAT
    )
    df['dist_to_gate_km'] = haversine_vectorized(
        df['lon'],
        df['lat'],
        GATE_LON,
        GATE_LAT
    )

    if 'route_s_km' not in df.columns or 'cross_track_km' not in df.columns:
        projection_result = [
            project_point_to_centerline(
                lon,
                lat,
                centerline_m,
                centerline_transformer
            )
            for lon, lat in zip(df['lon'], df['lat'])
        ]
        projection_result = pd.DataFrame(
            projection_result,
            columns=['route_s_km', 'cross_track_km'],
            index=df.index
        )
        df = pd.concat([df, projection_result], axis=1)
    else:
        df['route_s_km'] = pd.to_numeric(df['route_s_km'], errors='coerce')
        df['cross_track_km'] = pd.to_numeric(df['cross_track_km'], errors='coerce')

    df['start_route_s_km'] = (
        df.groupby('voyage_id')['route_s_km']
          .transform('first')
    )
    df['route_traveled_km'] = df['route_s_km'] - df['start_route_s_km']
    df['route_traveled_km_clipped'] = df['route_traveled_km'].clip(lower=0)

    gate_x, gate_y = centerline_transformer.transform(GATE_LON, GATE_LAT)
    gate_route_s_km = centerline_m.project(Point(gate_x, gate_y)) / 1000.0
    df['route_remaining_km'] = (
        gate_route_s_km - df['route_s_km']
    ).clip(lower=0)

    df = add_time_rolling_sog(df)

    df['avg_speed_overall'] = (
        df['dist_traveled_km'] / (df['pure_hours_elapsed'] + 1e-5)
    )
    df['avg_route_speed_knots'] = (
        df['route_traveled_km_clipped']
        / (df['pure_hours_elapsed'] + 1e-5)
        / 1.852
    )

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df


def to_local_time_string(series):
    """把时间序列转换为北京时间字符串。"""
    return (
        pd.to_datetime(series, utc=True, errors='coerce')
        .dt.tz_convert(LOCAL_TZ)
        .dt.strftime('%Y-%m-%d %H:%M:%S')
    )


def export_daily_prediction_files(result, data_date):
    """按预测到达日期导出当日和次日到达船舶列表。"""
    if result.empty:
        base_date = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
    else:
        upload_local = pd.to_datetime(
            result['prediction_upload_time'],
            errors='coerce'
        ).dt.tz_localize(LOCAL_TZ)
        base_date = upload_local.max().normalize()

    output_cols = [
        'mmsi',
        'vessel_name',
        'prediction_upload_time',
        'pred_sailing_hours',
        'pred_arrival_time',
    ]

    arrival_local = pd.to_datetime(
        result['pred_arrival_time'],
        errors='coerce'
    ).dt.tz_localize(LOCAL_TZ)

    for offset in [0, 1]:
        day_start = base_date + pd.Timedelta(days=offset)
        day_end = day_start + pd.Timedelta(days=1)
        day_text = day_start.strftime('%Y-%m-%d')
        output_file = PRED_DATA_DIR / DAILY_OUTPUT_TEMPLATE.format(data_date  = data_date, date=day_text)

        day_mask = (arrival_local >= day_start) & (arrival_local < day_end)
        day_df = result.loc[day_mask, output_cols].copy()
        day_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(
            f"{day_text} 预计到达船舶: {len(day_df)} 条，"
            f"已导出至: {output_file}"
        )


def main():
    global PRED_DATA_DIR, INPUT_FILE, OUTPUT_ALL_FILE
    args = parse_args()
    paths = configure_paths(args.data_date)
    PRED_DATA_DIR = paths["pred_data_dir"]
    INPUT_FILE = paths["input_filtered_file"]
    OUTPUT_ALL_FILE = paths["prediction_all_file"]


    PRED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_FILE.exists() or not FEATURE_META_FILE.exists():
        raise FileNotFoundError(
            '未找到 xgboost_v8 模型文件。请先运行 non_stop_v8.py，'
            '生成 xgboost_v8_model.json 和 xgboost_v8_features.json。'
        )

    feature_meta = json.loads(
        FEATURE_META_FILE.read_text(encoding='utf-8')
    )
    features = feature_meta['features']
    ship_type_features = feature_meta.get('ship_type_features', [])

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    ais_df = load_filtered_input()
    if ais_df.empty:
        empty_cols = [
            'mmsi',
            'vessel_name',
            'prediction_upload_time',
            'pred_sailing_hours',
            'pred_arrival_time',
        ]
        empty_result = pd.DataFrame(columns=empty_cols)
        empty_result.to_csv(OUTPUT_ALL_FILE, index=False, encoding='utf-8-sig')
        export_daily_prediction_files(empty_result, args.data_date)
        print("输入数据为空，已导出空预测结果。")
        return

    feature_df = build_model_features(
        ais_df,
        features,
        ship_type_features
    )

    latest_df = (
        feature_df
        .sort_values(['voyage_id', 'postime'])
        .groupby('voyage_id', as_index=False)
        .tail(1)
        .copy()
    )

    X_live = latest_df[features]
    pred_sailing_hours = model.predict(X_live)
    pred_sailing_hours = np.clip(pred_sailing_hours, 0, None)

    pred_arrival_time_utc = (
        latest_df['postime']
        + pd.to_timedelta(pred_sailing_hours, unit='h')
    )

    result = pd.DataFrame({
        'mmsi': latest_df['mmsi'].to_numpy(),
        'prediction_upload_time': to_local_time_string(latest_df['postime']),
        'pred_sailing_hours': pred_sailing_hours,
        'pred_arrival_time': to_local_time_string(pred_arrival_time_utc),
    })
    vessel_name_map = load_vessel_name_map(VESSEL_NAME_FILE)
    result.insert(
        1,
        'vessel_name',
        result['mmsi'].map(vessel_name_map)
    )
    result = result.sort_values('pred_arrival_time').reset_index(drop=True)

    result.to_csv(OUTPUT_ALL_FILE, index=False, encoding='utf-8-sig')
    export_daily_prediction_files(result, args.data_date)

    print(f"已完成 {len(result)} 条船舶 ETA 预测")
    print(
        f"船名映射成功: {result['vessel_name'].notna().sum()}/{len(result)} 条"
    )
    print(f"全量预测结果已导出至: {OUTPUT_ALL_FILE}")
    print(result.head(30).to_string(index=False))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Для линии регрессии
import statsmodels.api as sm # Для STL декомпозиции
import plotly.express as px # Для интерактивных графиков
import folium # Для интерактивных карт
import webbrowser # Для открытия HTML файлов
import os # Для работы с путями файлов

# --- Конфигурация ---
# Путь к вашему CSV-файлу. Замените, если файл находится в другом месте.
CSV_FILE_PATH = "openaq_location_282706_measurments.csv"

# --- Модуль загрузки данных из CSV ---

def load_data_from_csv(file_path, parameters_to_extract):
    """
    Загружает и предварительно обрабатывает данные из CSV-файла.

    Args:
        file_path (str): Путь к CSV-файлу.
        parameters_to_extract (list): Список строковых имен параметров для извлечения (например, ['pm25', 'pm10']).

    Returns:
        tuple: (pandas.DataFrame, dict)
               - DataFrame с данными, где индекс - datetime_utc, а столбцы - параметры.
               - Словарь с метаданными (latitude, longitude, location_name).
               Возвращает (None, None) в случае ошибки.
    """
    print(f"Загрузка данных из CSV-файла: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"CSV-файл успешно прочитан. Обнаружено {len(df)} строк.")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {file_path}")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении CSV-файла: {e}")
        return None, None

    # Проверка наличия необходимых столбцов
    required_cols = ['datetimeUtc', 'parameter', 'value']
    for col in required_cols:
        if col not in df.columns:
            print(f"Ошибка: Отсутствует необходимый столбец '{col}' в CSV-файле.")
            return None, None

    # 1. Преобразование и очистка дат
    df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'], errors='coerce')
    df.dropna(subset=['datetimeUtc'], inplace=True)
    if df.empty:
        print("Ошибка: Нет валидных дат после преобразования в CSV.")
        return None, None

    # 2. Фильтрация по нужным параметрам
    df_filtered = df[df['parameter'].isin(parameters_to_extract)].copy() # Используем .copy() для избежания SettingWithCopyWarning
    if df_filtered.empty:
        print(f"Предупреждение: Параметры {parameters_to_extract} не найдены в CSV-файле.")
        # Возвращаем пустой DataFrame, но с метаданными, если они есть
        meta = {
            'latitude': df['latitude'].iloc[0] if 'latitude' in df.columns and not df.empty else None,
            'longitude': df['longitude'].iloc[0] if 'longitude' in df.columns and not df.empty else None,
            'name': df['location_name'].iloc[0] if 'location_name' in df.columns and not df.empty else "Unknown Location"
        }
        return pd.DataFrame(index=pd.to_datetime([])), meta


    # 3. Преобразование значений в числовой тип
    df_filtered.loc[:, 'value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    df_filtered.dropna(subset=['value'], inplace=True) # Удаляем строки, где значение не удалось преобразовать

    # 4. Удаление отрицательных значений
    df_filtered = df_filtered[df_filtered['value'] >= 0]

    if df_filtered.empty:
        print("Предупреждение: Нет валидных данных после начальной очистки и фильтрации.")
        meta = {
            'latitude': df['latitude'].iloc[0] if 'latitude' in df.columns and not df.empty else None,
            'longitude': df['longitude'].iloc[0] if 'longitude' in df.columns and not df.empty else None,
            'name': df['location_name'].iloc[0] if 'location_name' in df.columns and not df.empty else "Unknown Location"
        }
        return pd.DataFrame(index=pd.to_datetime([])), meta


    # 5. Извлечение метаданных (берем из первой строки отфильтрованных данных)
    metadata = {
        'latitude': df_filtered['latitude'].iloc[0] if 'latitude' in df_filtered.columns else None,
        'longitude': df_filtered['longitude'].iloc[0] if 'longitude' in df_filtered.columns else None,
        'name': df_filtered['location_name'].iloc[0] if 'location_name' in df_filtered.columns else "Unknown Location",
        'unit': {} # Словарь для хранения единиц по каждому параметру
    }

    # 6. Поворот таблицы (pivot) для получения параметров в виде столбцов
    try:
        data_pivot = df_filtered.pivot_table(index='datetimeUtc', columns='parameter', values='value')
    except Exception as e:
        print(f"Ошибка при повороте таблицы: {e}")
        # Попытка удалить дубликаты перед поворотом, если это причина
        print("Попытка удалить дубликаты по (datetimeUtc, parameter) и повторить поворот...")
        df_filtered_dedup = df_filtered.drop_duplicates(subset=['datetimeUtc', 'parameter'], keep='first')
        try:
            data_pivot = df_filtered_dedup.pivot_table(index='datetimeUtc', columns='parameter', values='value')
        except Exception as e_pivot_again:
            print(f"Повторная ошибка при повороте таблицы: {e_pivot_again}")
            return None, None


    # 7. Установка и сортировка индекса (уже сделано datetimeUtc, но на всякий случай)
    data_pivot.index = pd.to_datetime(data_pivot.index)
    data_pivot.sort_index(inplace=True)

    # 8. Извлечение единиц измерения для каждого параметра
    for param in parameters_to_extract:
        if param in data_pivot.columns:
            # Находим первую не-NaN запись для этого параметра в исходном отфильтрованном df
            param_specific_df = df_filtered[df_filtered['parameter'] == param]
            if not param_specific_df.empty and 'unit' in param_specific_df.columns:
                metadata['unit'][param] = param_specific_df['unit'].iloc[0]
            else:
                metadata['unit'][param] = 'N/A'

    print(f"Данные успешно загружены и преобразованы. Форма DataFrame: {data_pivot.shape}")
    return data_pivot, metadata

# --- Модуль анализа данных ---

def calculate_basic_stats(df, parameters):
    """ Рассчитывает базовые статистики для каждого параметра в списке. """
    if df is None or df.empty: print("DataFrame пуст, расчет статистик невозможен."); return None
    print("\n--- Расчет базовых статистик ---")
    all_stats = {}
    for param in parameters:
        if param in df.columns:
            print(f"\nСтатистики для '{param}':")
            if df[param].notna().any():
                stats = df[param].describe()
                stats['median'] = df[param].median()
                stats['25%'] = df[param].quantile(0.25)
                stats['75%'] = df[param].quantile(0.75)
                print(stats)
                all_stats[param] = stats
            else:
                print(f"Столбец '{param}' не содержит валидных данных для расчета статистик.")
        else:
            print(f"Предупреждение: Столбец '{param}' не найден в DataFrame для расчета статистик.")
    return all_stats

def calculate_moving_average(df, parameters, window_days=7):
    """ Рассчитывает скользящее среднее для каждого параметра в списке. """
    if df is None or df.empty: print("DataFrame пуст, расчет скользящего среднего невозможен."); return None
    print(f"\n--- Расчет {window_days}-дневного скользящего среднего ---")
    df_ma = pd.DataFrame(index=df.index)
    df_sorted = df.sort_index()
    for param in parameters:
        if param in df_sorted.columns:
            if df_sorted[param].notna().any():
                print(f"Расчет для '{param}'...")
                df_ma[f'{param}_ma_{window_days}d'] = df_sorted[param].rolling(window=f'{window_days}D', min_periods=1, closed='right').mean().copy()
            else:
                 print(f"Столбец '{param}' не содержит валидных данных для расчета скользящего среднего.")
        else:
            print(f"Предупреждение: Столбец '{param}' не найден для расчета скользящего среднего.")
    return df_ma

def perform_stl_decomposition(df, parameters, period=None):
    """ Выполняет STL декомпозицию для каждого параметра в списке. """
    if df is None or df.empty: print("DataFrame пуст, STL декомпозиция невозможна."); return None
    print("\n--- Выполнение STL декомпозиции ---")
    stl_results = {}
    for param in parameters:
        if param in df.columns:
            print(f"\nSTL для '{param}':")
            series = df[param].copy().interpolate(method='linear').dropna()

            if series.empty:
                print("  Ошибка STL: Временной ряд пуст после обработки пропусков.")
                continue

            required_length = 2 * (period if period is not None and period > 1 else 2)
            if len(series) < required_length:
                print(f"  Ошибка STL: Временной ряд слишком короткий ({len(series)} точек, требуется >= {required_length}).")
                continue
            if period is not None and period >= len(series) / 2:
                 print(f"  Ошибка STL: Период ({period}) должен быть меньше половины длины ряда ({len(series)}).")
                 continue

            try:
                current_period = period if period is not None and period > 1 else None
                print(f"  Выполнение STL с периодом: {current_period} для {len(series)} точек")
                if not series.index.is_monotonic_increasing:
                     print("  Предупреждение: Индекс временного ряда не монотонно возрастает. Сортировка...")
                     series = series.sort_index()
                if series.index.has_duplicates:
                     print("  Предупреждение: Обнаружены дубликаты в индексе временного ряда после интерполяции. Агрегирование...")
                     series = series.groupby(series.index).mean()
                     print(f"  Размер ряда после агрегации дубликатов: {len(series)}")
                     if len(series) < required_length:
                          print(f"  Ошибка STL: Временной ряд слишком короткий ({len(series)} точек) после агрегации дубликатов.")
                          continue

                if len(series.index.unique()) < required_length:
                     print(f"  Ошибка STL: Недостаточно уникальных временных точек ({len(series.index.unique())}) после обработки пропусков и агрегации.")
                     continue

                stl = sm.tsa.STL(series, period=current_period, robust=True)
                result = stl.fit()
                print("  STL декомпозиция успешно выполнена.")
                stl_results[param] = result
            except Exception as e:
                print(f"  Ошибка при выполнении STL декомпозиции для '{param}': {e}")
        else:
            print(f"Предупреждение: Столбец '{param}' не найден для STL декомпозиции.")
    return stl_results

def calculate_and_plot_correlation(df, param1, param2, metadata, filename="correlation_plot.png"):
    """
    Рассчитывает и визуализирует корреляцию между двумя параметрами.

    Args:
        df (pandas.DataFrame): DataFrame с данными (индекс - datetime, столбцы - параметры).
        param1 (str): Имя первого параметра.
        param2 (str): Имя второго параметра.
        metadata (dict): Словарь с метаданными, включая единицы измерения.
        filename (str): Имя файла для сохранения графика.
    """
    if param1 in df.columns and param2 in df.columns:
        # Удаляем строки, где любое из значений NaN для корректного расчета
        df_corr = df[[param1, param2]].dropna()

        if len(df_corr) < 2: # Нужно как минимум 2 точки для корреляции
            print(f"Недостаточно данных для расчета корреляции между {param1} и {param2} после удаления NaN.")
            return

        correlation = df_corr[param1].corr(df_corr[param2])
        print(f"\nКорреляция между {param1} и {param2}: {correlation:.4f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(df_corr[param1], df_corr[param2], alpha=0.5, label=f'Данные ({len(df_corr)} точек)')

        # Добавление линии регрессии
        try:
            # Проверка на наличие достаточного разброса данных для polyfit
            if len(df_corr[param1].unique()) > 1 and len(df_corr[param2].unique()) > 1 :
                m, b = np.polyfit(df_corr[param1], df_corr[param2], 1)
                plt.plot(df_corr[param1], m * df_corr[param1] + b, color='red', label=f'Линия регрессии\ny={m:.2f}x+{b:.2f}\nR²={correlation**2:.2f}')
            else:
                print("Недостаточный разброс данных для построения линии регрессии.")
        except Exception as e:
            print(f"Не удалось построить линию регрессии: {e}")

        unit1 = metadata.get('unit', {}).get(param1, 'units')
        unit2 = metadata.get('unit', {}).get(param2, 'units')

        plt.title(f'Корреляция и диаграмма рассеяния: {param1} vs {param2}')
        plt.xlabel(f'{param1} ({unit1})')
        plt.ylabel(f'{param2} ({unit2})')
        plt.legend()
        plt.grid(True)

        try:
            plt.savefig(filename)
            print(f"График корреляции сохранен в: {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении графика корреляции: {e}")
        plt.show()
        plt.close()
    else:
        missing_params = []
        if param1 not in df.columns: missing_params.append(param1)
        if param2 not in df.columns: missing_params.append(param2)
        print(f"Параметр(ы) {', '.join(missing_params)} отсутствуют в DataFrame для расчета корреляции.")


# --- Модуль визуализации ---

def plot_time_series(df, parameters, metadata, moving_averages=None, filename_prefix='timeseries_mpl'):
    """ Строит временные ряды для каждого параметра (Matplotlib). """
    if df is None or df.empty: print("Невозможно построить график Matplotlib: DataFrame пуст."); return
    print("\n--- Построение графиков временных рядов (Matplotlib) ---")

    for param in parameters:
        if param in df.columns and df[param].notna().any():
            unit = metadata.get('unit', {}).get(param, 'units')
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[param], label=f'Измерения {param}', alpha=0.7, marker='.', markersize=3, linestyle='-')

            ma_col_name = f'{param}_ma_7d' # TODO: Сделать имя колонки MA динамическим
            if moving_averages is not None and ma_col_name in moving_averages.columns and moving_averages[ma_col_name].notna().any():
                 plt.plot(moving_averages.index, moving_averages[ma_col_name], label=f'Скользящее среднее 7d', color='red', linewidth=2)

            plt.title(f'Временной ряд концентрации {param} ({metadata.get("name", "")})')
            plt.xlabel('Дата и время (UTC)')
            plt.ylabel(f'Концентрация ({unit})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = f"{filename_prefix}_{param}.png"
            try:
                 plt.savefig(filename)
                 print(f"График для '{param}' сохранен в файл: {filename}")
            except Exception as e:
                 print(f"Ошибка при сохранении графика для '{param}' в файл {filename}: {e}")
            plt.close()
        elif param not in df.columns:
            print(f"Предупреждение: Столбец '{param}' не найден для построения графика.")
        else:
            print(f"Предупреждение: Нет данных для построения графика для параметра '{param}'.")


def plot_histogram(df, parameters, metadata, bins=30, filename_prefix='histogram_mpl'):
    """ Строит гистограммы для каждого параметра (Matplotlib). """
    if df is None or df.empty: print("Невозможно построить гистограмму Matplotlib: DataFrame пуст."); return
    print("\n--- Построение гистограмм (Matplotlib) ---")
    for param in parameters:
        if param in df.columns and df[param].notna().any():
            unit = metadata.get('unit', {}).get(param, 'units')
            plt.figure(figsize=(10, 6))
            plt.hist(df[param].dropna(), bins=bins, edgecolor='black')
            plt.title(f'Гистограмма распределения {param} ({metadata.get("name", "")})')
            plt.xlabel(f'Концентрация ({unit})')
            plt.ylabel('Частота')
            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()
            filename = f"{filename_prefix}_{param}.png"
            try:
                 plt.savefig(filename)
                 print(f"Гистограмма для '{param}' сохранена в файл: {filename}")
            except Exception as e:
                 print(f"Ошибка при сохранении гистограммы для '{param}' в файл {filename}: {e}")
            plt.close()
        elif param not in df.columns:
            print(f"Предупреждение: Столбец '{param}' не найден для построения гистограммы.")
        else:
             print(f"Предупреждение: Нет данных для построения гистограммы для параметра '{param}'.")

def plot_box_plot(df, parameters, metadata, filename_prefix='boxplot_mpl'):
    """ Строит ящики с усами для каждого параметра (Matplotlib). """
    if df is None or df.empty: print("Невозможно построить ящик с усами: DataFrame пуст."); return
    print("\n--- Построение ящиков с усами (Matplotlib) ---")
    for param in parameters:
        if param in df.columns and df[param].notna().any():
            unit = metadata.get('unit', {}).get(param, 'units')
            plt.figure(figsize=(8, 6))
            plt.boxplot(df[param].dropna(), vert=True, patch_artist=True, showfliers=True)
            plt.title(f'Ящик с усами для {param} ({metadata.get("name", "")})')
            plt.ylabel(f'Концентрация ({unit})')
            plt.xticks([1], [param])
            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()
            filename = f"{filename_prefix}_{param}.png"
            try:
                 plt.savefig(filename)
                 print(f"Ящик с усами для '{param}' сохранен в файл: {filename}")
            except Exception as e:
                 print(f"Ошибка при сохранении ящика с усами для '{param}' в файл {filename}: {e}")
            plt.close()
        elif param not in df.columns:
            print(f"Предупреждение: Столбец '{param}' не найден для построения ящика с усами.")
        else:
             print(f"Предупреждение: Нет данных для построения ящика с усами для параметра '{param}'.")


def plot_stl_decomposition(stl_results, metadata, filename_prefix='stl_decomposition'):
    """ Визуализирует результаты STL для каждого параметра (Matplotlib). """
    if not stl_results: print("Нет результатов STL для визуализации."); return
    print("\n--- Визуализация результатов STL декомпозиции ---")
    for param, result in stl_results.items():
        print(f"Визуализация STL для '{param}'...")
        try:
            fig = result.plot()
            fig.set_size_inches(12, 8)
            fig.suptitle(f'STL Декомпозиция для {param} ({metadata.get("name", "")})', y=1.02)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = f"{filename_prefix}_{param}.png"
            plt.savefig(filename)
            print(f"График STL для '{param}' сохранен в файл: {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"Ошибка при визуализации STL для '{param}': {e}")


def plot_interactive_time_series(df, parameters, metadata, title_prefix='Интерактивный временной ряд', filename_prefix='interactive_timeseries'):
    """ Строит интерактивные временные ряды для каждого параметра (Plotly). """
    if df is None or df.empty: print("Невозможно построить интерактивный график Plotly: DataFrame пуст."); return
    print("\n--- Создание интерактивных графиков (Plotly) ---")
    df_reset = df.reset_index() # Нужен datetimeUtc как столбец для Plotly

    for param in parameters:
        if param in df.columns and df[param].notna().any():
            unit = metadata.get('unit', {}).get(param, 'units')
            print(f"Создание графика для '{param}'...")
            try:
                 fig = px.line(df_reset, x='datetimeUtc', y=param, title=f'{title_prefix} {param} ({metadata.get("name", "")})',
                               labels={'datetimeUtc': 'Дата и время (UTC)', param: f'Концентрация ({unit})'})
                 fig.update_layout(xaxis_title='Дата и время (UTC)', yaxis_title=f'Концентрация ({unit})')
                 filename = f"{filename_prefix}_{param}.html"
                 fig.write_html(filename)
                 print(f"Интерактивный график для '{param}' сохранен в: {filename}")
                 webbrowser.open('file://' + os.path.realpath(filename))
            except Exception as e:
                 print(f"Ошибка при создании или открытии интерактивного графика для '{param}': {e}")
        elif param not in df.columns:
             print(f"Предупреждение: Столбец '{param}' не найден для построения интерактивного графика.")
        else:
             print(f"Предупреждение: Нет данных для построения интерактивного графика для параметра '{param}'.")


def plot_locations_map(metadata, filename='location_map.html'):
    """ Отображает точку на интерактивной карте Folium, используя метаданные. """
    if not metadata or metadata.get('latitude') is None or metadata.get('longitude') is None:
        print("Нет данных о координатах для отображения на карте."); return
    print("\n--- Создание интерактивной карты локации (Folium) ---")

    lat = metadata['latitude']
    lon = metadata['longitude']
    location_name = metadata.get('name', 'N/A')

    m = folium.Map(location=[lat, lon], zoom_start=12)

    popup_text = f"<b>{location_name}</b><br>" \
                 f"Координаты: ({lat:.4f}, {lon:.4f})"
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=location_name
    ).add_to(m)

    try:
         m.save(filename)
         print(f"Интерактивная карта сохранена в: {filename}")
         webbrowser.open('file://' + os.path.realpath(filename))
    except Exception as e:
         print(f"Ошибка при сохранении или открытии карты: {e}")

# --- Основной блок выполнения (из CSV) ---
if __name__ == "__main__":
    # Параметры, которые мы хотим проанализировать из CSV
    parameters_to_analyze = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']

    # Загрузка данных из CSV
    air_quality_data, location_metadata = load_data_from_csv(CSV_FILE_PATH, parameters_to_analyze)

    # --- Анализ и Визуализация (если данные есть) ---
    if air_quality_data is not None and not air_quality_data.empty:
        # Определяем, какие параметры реально были загружены и имеют данные
        loaded_parameters = [p for p in parameters_to_analyze if p in air_quality_data.columns and air_quality_data[p].notna().any()]

        if not loaded_parameters:
             print("\nВ загруженных данных нет валидных столбцов для анализа.")
        else:
             print(f"\n--- Анализ данных для параметров: {loaded_parameters} ---")
             location_name_for_title = location_metadata.get("name", "Unknown Location")

             # 1. Базовые статистики
             calculate_basic_stats(air_quality_data, loaded_parameters)

             # 2. Скользящее среднее
             moving_averages = calculate_moving_average(air_quality_data, loaded_parameters, window_days=7)

             # 3. STL Декомпозиция
             stl_period = None
             try:
                 if isinstance(air_quality_data.index, pd.DatetimeIndex):
                      if len(air_quality_data.index) > 2: # Нужно хотя бы несколько точек
                           inferred_freq = pd.infer_freq(air_quality_data.index)
                           print(f"Определенная частота индекса: {inferred_freq}")
                           if inferred_freq and ('H' in inferred_freq or 'T' in inferred_freq): stl_period = 24 * 7 # Недельная для часовых/минутных
                           elif inferred_freq and 'D' in inferred_freq: stl_period = 7 # Недельная для дневных
                           else: # Если частота не определена, пробуем по медианной разнице
                                time_diff = air_quality_data.index.to_series().diff().median()
                                if not pd.isna(time_diff):
                                     if time_diff <= pd.Timedelta(hours=1.5): stl_period = 24 * 7
                                     elif time_diff <= pd.Timedelta(days=1.5): stl_period = 7
                                     else: print(f"Не удалось определить частоту данных для STL (медианная разница: {time_diff}).")
                                else:
                                     print("Не удалось рассчитать медианную разницу времени для определения частоты.")
                      else:
                           print("Недостаточно данных для определения частоты индекса.")
                 else:
                      print("Индекс DataFrame не является DatetimeIndex, не могу определить частоту.")
                 if stl_period: print(f"Предполагаемый период для STL: {stl_period}")
             except Exception as e:
                 print(f"Ошибка при определении периода STL: {e}")

             stl_results = perform_stl_decomposition(air_quality_data, loaded_parameters, period=stl_period)

             # 4. Анализ корреляции (между PM2.5 и PM10, если они есть)
             if 'pm25' in loaded_parameters and 'pm10' in loaded_parameters:
                 calculate_and_plot_correlation(air_quality_data, 'pm25', 'pm10', location_metadata,
                                                filename=f"correlation_pm25_pm10_{location_name_for_title.replace(' ', '_')}.png")
             else:
                 print("\nPM2.5 и/или PM10 отсутствуют в данных, корреляционный анализ не выполнен.")


             print(f"\n--- Визуализация данных для параметров: {loaded_parameters} ---")
             file_prefix = location_name_for_title.replace(' ', '_').lower()


             # 5. Статические графики (Matplotlib)
             plot_time_series(air_quality_data, loaded_parameters, location_metadata, moving_averages=moving_averages, filename_prefix=f"{file_prefix}_timeseries")
             plot_histogram(air_quality_data, loaded_parameters, location_metadata, filename_prefix=f"{file_prefix}_histogram")
             plot_box_plot(air_quality_data, loaded_parameters, location_metadata, filename_prefix=f"{file_prefix}_boxplot")

             # 6. График STL декомпозиции (Matplotlib)
             if stl_results:
                 plot_stl_decomposition(stl_results, location_metadata, filename_prefix=f"{file_prefix}_stl")

             # 7. Интерактивный график (Plotly)
             plot_interactive_time_series(air_quality_data, loaded_parameters, location_metadata,
                                          title_prefix=f'Интерактивный ряд',
                                          filename_prefix=f"{file_prefix}_interactive")

             # 8. Карта локации (Folium)
             if location_metadata.get('latitude') is not None and location_metadata.get('longitude') is not None:
                 plot_locations_map(location_metadata, filename=f"{file_prefix}_location_map.html")
             else:
                 print("Координаты для карты отсутствуют в метаданных.")

    elif air_quality_data is not None and air_quality_data.empty:
         print("\nDataFrame пуст после загрузки и начальной обработки. Анализ и визуализация невозможны.")
    else: # air_quality_data is None
        print("\nНе удалось загрузить или обработать данные из CSV. Анализ и визуализация невозможны.")

    print("\n--- Прототип завершил работу ---")


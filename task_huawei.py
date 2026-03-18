!wget -O trace.csv.gz https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureLMMInferenceTrace_multimodal.csv.gz


import heapq
import math
from collections import deque
from typing import List, Deque, Optional

import numpy as np
import pandas as pd

# ================================
'''
Ниже представлен код симуляции работы программы, которая на основе логов подбирает 
необходимое количество ГПУ для работы с такими запросами. 
В этом файле сосредочился
на корректном написании кода, его оформлении, описании работы функций, типизации.
В файле min_gpu_for_azure_logs.ipynb код ниже будет переиспользоваться
и в том же файле будет ответ на задачу сколько гпу + аналитика логов и подбор параметров
'''
# =================================

# gzip по расширению
df = pd.read_csv("trace.csv.gz", compression="gzip")


# Глобальные константы. взяли всего одну GPU,

MEM_M = 80000  # 80 gb memory
MEM_MODEL_TEXT_EXTRACTION = 600  # MB  weights для самой модели детекцим
MEM_M -= MEM_MODEL_TEXT_EXTRACTION

MEM_X = 10  # вес самой картинки 1024px
MEM_Y = 1  # мб на 1 токен контекста (входной текст)
MEM_Z = 1  # мб на 1 сгенерированный токен ( выходной текст)
# ТОКЕНЫ контекста и генерации разные вещи и хранятся в VRAM

# compute_costs на препроцессинг картинки, обработку токена, время генерации 1 токена
COST_A = 0.5  # на картинку
COST_B = 0.01  # обработка 1 токена контекста
COST_C = 0.05  # 1 сгенерированный токен

# limitations global contants
LIM_TTFT = 10  # P ttft not more P
LIM_GEN_NEXT_TOK = 1  # D limit time stage 3 0.05 это sec/token

MAX_CHUNK_SIZE = 512  # tokens per forward pass # 512
MAX_BATCH_SIZE = 20  # сколько в батч задвинем реквестов


class Request:
    """
    Поля: id, время_прибытия, кол_во_картинок, токены_контекста, токены_генерации
    То что содержится в классе:
    id - запроса
    start_processing_time - время когда gpu только НАЧАЛА работать
    ttft_time - закончился препроцессинг картинок и контекста
    finish_time - время сгенерированного последнего токена, который должен <= T (по ТЗ)
    limit_failed - флаг нарушение одно из требований по тз
    time_stage_3 - время до последнего токена
    max_gen_tokens - сколько у самого жирного реквеста будет токенов для генерации в батче
    request берез из df[['Arrival_Sec', 'ContextTokens', 'NumImages', 'GeneratedTokens']]

    """

    def __init__(self, id: int, time_to_come: float, token_context: int, num_images: int, token_generation: int):
        self.id = id
        self.time_to_come = time_to_come
        self.num_images = num_images
        self.token_context = token_context
        self.token_generation = token_generation

        # данные симуляции
        self.start_processing_time: Optional[float] = None
        self.ttft_time: Optional[float] = None

        self.images_left = self.num_images
        self.images_processed = False  # Флаг обработки изображений
        self.context_tokens_left = self.token_context
        self.tokens_left_to_generate = self.token_generation  # сколько осталось генерить
        self.allocated_vram = 0  # сколько памяти запрос схавал

        self.finish_time: Optional[float] = None
        self.limit_failed = False
        self.time_stage_3: Optional[float] = None
        self.fail_reason: Optional[str] = None

    def get_memory_per_request(self) -> int:
        """
        сколько инпут займет памяти VRAM
        для картинки, контекстов, генераций
        одного реквеста
        """
        memory_images = self.num_images * MEM_X
        memory_context = self.token_context * MEM_Y
        memory_generation = self.token_generation * MEM_Z
        return memory_images + memory_context + memory_generation


class Accelerator:
    """
    класс GPU c характеристиками
    статус (Свободен/Занят), доступная_память, вычислительная_мощность, номер_гпу
    """

    def __init__(self, status: str, memory: int, computing: int, gpu_id: int):
        self.status = status
        self.free_memory = MEM_M
        self.computing = computing
        self.active_batch: List[Request] = []  # наша VRAM
        self.is_ticking = False  # это флаг того, что если загрузится гпу, то она в работе
        self.gpu_id = gpu_id
        self.total_compute_time = 0.0


class Event:
    """
    класс события
    Поля: время_события, тип_события
    data - весь экземляр класса Accelerator
    list_current_batch - лист всех нынешнего батча
    """

    def __init__(self, time_event: float, type_event: str, data, list_current_batch=None):
        self.time_event = time_event
        self.type_event = type_event
        self.data = data
        self.list_current_batch = list_current_batch

    # для сравнения событий через магическиий метода __less than__
    def __lt__(self, other) -> bool:
        return self.time_event < other.time_event


def make_heap_from_df(df: pd.DataFrame) -> List[Event]:

    scheduler_event = []
    # берем сырой df
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    # Берем время самого первого запроса
    start_time = df["TIMESTAMP"].min()

    # Считаем разницу в секундах (с долями миллисекунд)
    df["Arrival_Sec"] = (df["TIMESTAMP"] - start_time).dt.total_seconds()
    # если не отсортированно изначально
    df.sort_values(by="TIMESTAMP", inplace=True)

    # из дата фрейма в инициализацию календаря
    for row in df[["Arrival_Sec", "ContextTokens", "NumImages", "GeneratedTokens"]].itertuples():
        req = Request(
            row.Index,
            row.Arrival_Sec,
            row.ContextTokens,
            row.NumImages,
            row.GeneratedTokens,
        )

        event = Event(req.time_to_come, "NEW REQUEST", req)
        scheduler_event.append(event)

    # подаем лист, делает все на месте, лист теперь куча
    heapq.heapify(scheduler_event)
    return scheduler_event


def check_limitations(req: Request) -> None:
    """
    проверка sla с подробным логированием
    """
    ttft_duration = req.ttft_time - req.time_to_come
    average_time_per_token = 0.0

    print(
        f"[LOG SLA CHECK] Req {req.id}: TTFT = {ttft_duration:.2f}s (Limit {LIM_TTFT}) | "
        f"Decode Avg = {average_time_per_token:.4f}s (Limit {LIM_GEN_NEXT_TOK}) | "
        f"Gen Tokens = {req.token_generation}"
    )
    if ttft_duration > LIM_TTFT:
        req.limit_failed = True
        req.fail_reason = "TTFT"
        print(f"   -----> [SLA FAILED] REQ {req.id}: ПРЕВЫШЕН ЛИМИТ TTFT!")
    if req.token_generation > 0:
        total_stage_3_time = req.finish_time - req.ttft_time
        average_time_per_token = total_stage_3_time / req.token_generation
        if average_time_per_token > LIM_GEN_NEXT_TOK:
            req.limit_failed = True
            req.fail_reason = "DECODE"
            print(
                f" ----> [SLA FAILED] REQ {req.id}: ПРЕВЫШЕН ЛИМИТ ГЕНЕРАЦИИ НА ТОКЕН"
            )

def collect_finished_requests(time_now: float, accelerator: Accelerator, completed_requests: List[Request]) -> None:
    """
    Очищает активный батч от полностью сгенерированных запросов и освобождает память.
    Переменные: time_now (текущее время), accelerator (текущая GPU), completed_requests (список готовых запросов).
    Идеи: Высвобождение KV-Cache. Как только запрос полностью сгенерирован, память немедленно возвращается в пул (free_memory) для других задач.
    """
    finished_requests = []

    for req in accelerator.active_batch:
        if req.tokens_left_to_generate == 0 and req.ttft_time is not None:
            req.finish_time = time_now
            req.time_stage_3 = req.finish_time - req.ttft_time
            accelerator.free_memory += req.allocated_vram
            check_limitations(req)
            finished_requests.append(req)

    finished_set = set(finished_requests)
    completed_requests.extend(finished_requests)
    accelerator.active_batch = [
        req for req in accelerator.active_batch if req not in finished_set
    ]


def admit_new_requests(deque_requests: Deque[Request], accelerator: Accelerator) -> None:
    """
    Забирает новые запросы из очереди ожидания в активный батч, если хватает VRAM.
    Переменные: deque_requests (очередь ожидания), accelerator (текущая GPU).
    Тут Continuous Batching. Запросы добавляются в батч динамически "на лету", не дожидаясь,
     пока все предыдущие запросы закончат окончательную генерацию.
    """
    while deque_requests:
        # Ограничение  батча потому что жадник барагозит
        if len(accelerator.active_batch) >= MAX_BATCH_SIZE:
            break

        next_req = deque_requests[0]
        tokens_for_first_chunk = min(MAX_CHUNK_SIZE, next_req.token_context)
        initial_mem_needed = (next_req.num_images * MEM_X) + (
            tokens_for_first_chunk * MEM_Y
        )

        if initial_mem_needed > MEM_M:
            deque_requests.popleft()
            continue

        if initial_mem_needed <= accelerator.free_memory:
            req = deque_requests.popleft()
            accelerator.active_batch.append(req)

            img_mem = req.images_left * MEM_X
            chunk_mem = tokens_for_first_chunk * MEM_Y

            total_reserved = img_mem + chunk_mem
            accelerator.free_memory -= total_reserved
            req.allocated_vram += total_reserved

            # Обнуляем картинки, так как память под них выделена
            req.images_left = 0
        else:
            break


def evict_requests_if_oom(req: Request, accelerator: Accelerator, deque_requests: Deque[Request]) -> None:
    """
    Вытесняет запросы обратно в очередь, если памяти для генерации следующего токена недостаточно.
    Переменные: req (текущий запрос декодера), accelerator (GPU), deque_requests (очередь ожидания).
    Eviction Policy  защита от OOM. При нехватке памяти запрос leftappend к нашему буфера (деке)
    """
    while accelerator.free_memory < MEM_Z:
        victim_req = max(
            accelerator.active_batch, key=lambda r: r.tokens_left_to_generate
        )
        if victim_req is req:
            break

        accelerator.free_memory += victim_req.allocated_vram
        victim_req.allocated_vram = 0
        victim_req.context_tokens_left = victim_req.token_context
        victim_req.ttft_time = None
        victim_req.images_processed = False 
        
        accelerator.active_batch.remove(victim_req)
        deque_requests.appendleft(victim_req)


def process_compute_step(time_now: float, deque_requests: Deque[Request], accelerator: Accelerator, scheduler_event: List[Event]) -> None:
    """
    Выполняет математику фаз Prefill и Decode, вычисляет длительность такта и планирует следующий тик.
    Переменные: time_now (текущее время), deque_requests (очередь), accelerator (GPU), scheduler_event (события).
    Chunked Prefill
    Заметил, что когда в батч к декодерам залетает жирный реквест на префилл, то батч
    начинает тормозить сильно. Поэтому в батч отправляется кусочек MAX_CHUNK_SIZE 
    который является параметром и я его вычисляю под определнный спек ГПУ в джупайтере
    """
    #  Распределение на Newbies (Prefill) и Decoders (Генерация)
    newbies = [i for i in accelerator.active_batch if i.ttft_time is None]
    # Защита от "путешествий во времени" (запрос становится декодером только когда настало его время)
    decoders = [k for k in accelerator.active_batch if k.ttft_time is not None]

    current_chunk_tokens = 0
    current_images_to_process = 0
    processed_decoders = 0

    # Обработка Prefill
    if newbies:
        for req in newbies:
            # Считаем картинки только один раз (когда контекст еще целый)
            if not req.images_processed:
                current_images_to_process += req.num_images
                req.images_processed = True

            #  берем остаток токенов, но не больше чанка
            tokens_to_compute = min(req.context_tokens_left, MAX_CHUNK_SIZE)
            current_chunk_tokens += tokens_to_compute
            # списываем токены, которые взяли в вычисление
            req.context_tokens_left -= tokens_to_compute

            if req.context_tokens_left > 0:
                next_chunk = min(req.context_tokens_left, MAX_CHUNK_SIZE)
                mem_needed = next_chunk * MEM_Y

                if accelerator.free_memory >= mem_needed:
                    req.context_tokens_left -= next_chunk
                    accelerator.free_memory -= mem_needed
                    req.allocated_vram += mem_needed

            if current_chunk_tokens >= MAX_CHUNK_SIZE:
                break

    # Обработка Decode EVICTION POLICY
    for req in decoders:
        # защита от утечки памяти
        if req not in accelerator.active_batch:
            continue

        evict_requests_if_oom(req, accelerator, deque_requests)

        if accelerator.free_memory >= MEM_Z:
            req.tokens_left_to_generate -= 1
            accelerator.free_memory -= MEM_Z
            req.allocated_vram += MEM_Z
            processed_decoders += 1

    time_images = (
        COST_A * math.sqrt(current_images_to_process)
        if current_images_to_process > 0
        else 0
    )
    time_prefill = (
        COST_B * math.sqrt(current_chunk_tokens) if current_chunk_tokens > 0 else 0
    )
    time_decode = (
        COST_C * math.sqrt(processed_decoders) if processed_decoders > 0 else 0
    )

    step_time = time_images + time_prefill + time_decode

    # предохранитель от Deadlock
    if step_time <= 0:
        step_time = 0.001

    # подсчет полезной работы
    accelerator.total_compute_time += step_time

    # Назначаем TTFT тем, кто закончил префилл
    for req in newbies:
        if req.context_tokens_left == 0 and req.ttft_time is None:
            req.ttft_time = time_now + step_time

    #  Создаем следующее событие (ТАКТ ВЫПОЛНЕН)
    this_event = Event(
        time_event=time_now + step_time, type_event="TICK_DONE", data=accelerator
    )
    heapq.heappush(scheduler_event, this_event)
    accelerator.is_ticking = True


def scheduler_step(
    time_now: float, 
    deque_requests: Deque[Request], 
    accelerator: Accelerator, 
    scheduler_event: List[Event], 
    completed_requests: List[Request]
) -> None:
    """
    Главный оркестратор (планировщик) симуляции. Управляет симуляцией запросов на GPU: 
    очищает завершенные, добирает новые из очереди и запускает вычисления такта.
    
    time_now - текущее время симуляции
    deque_requests - очередь ожидающих запросов
    accelerator - экземпляр текущей видеокарты
    scheduler_event - куча (heapq) будущих событий
    completed_requests - список успешно выполненных запросов
    """
    
    #  Очистка готовых
    collect_finished_requests(time_now, accelerator, completed_requests)
    
    # Добор новых 
    admit_new_requests(deque_requests, accelerator)

    # Если батч пуст - выключаем GPU
    if not accelerator.active_batch:
        accelerator.is_ticking = False
        return

    # 3. Управление памяти, расчет шага времени и планирование тика
    process_compute_step(time_now, deque_requests, accelerator, scheduler_event)



def simulate(N: int, df: pd.DataFrame):
    """
    df - наши логи
    N - количество гпу
    """
    time_now = 0
    list_accelerators = [Accelerator("FREE ACCELERATOR", MEM_M, 1, i) for i in range(N)]

    deque_requests = deque()
    completed_requests = []  # успешно завершенные запросы

    # взяли и внесли в кучу запросы
    scheduler_event = make_heap_from_df(df)

    while scheduler_event:
        this_event = heapq.heappop(scheduler_event)
        time_now = this_event.time_event

        if this_event.type_event == "NEW REQUEST":
            print(
                f"\n[LOG TIME: {time_now:.2f}] Новый запрос в систему: Req {this_event.data.id} (INPUT_TOK: {this_event.data.token_context}, GEN_TOK: {this_event.data.token_generation})"
            )
            deque_requests.append(this_event.data)
            # Убрали цикл for acc in list_accelerators отсюда, перенесли вниз

        elif this_event.type_event == "TICK_DONE":
            this_accelerator = this_event.data
            scheduler_step(
                time_now,
                deque_requests,
                this_accelerator,
                scheduler_event,
                completed_requests,
            )

        # УНИВЕРСАЛЬНЫЙ БУДИЛЬНИК ДЛЯ GPU:
        # Если в очереди есть задачи, пытаемся скормить их всем свободным картам
        if deque_requests:
            for acc in list_accelerators:
                if not acc.is_ticking:
                    if len(deque_requests) > 0:
                        print(
                            f"[LOG TIME: {time_now:.2f}] Будильник активировал свободную GPU {acc.gpu_id}"
                        )
                        scheduler_step(
                            time_now,
                            deque_requests,
                            acc,
                            scheduler_event,
                            completed_requests,
                        )

    return True, completed_requests, list_accelerators



"""Теперь сбор статистики с симуляции, где ПОСЧИТАНО где и что справилось неверно"""
#  либо кусочек
M = 500
df_ = df.head(M)
# либо весь дф
# df_ = df 

find_necessary_N = 0
success_requests = []


def run_analytics(N: int, df_sample: pd.DataFrame) -> None:
    '''
    тут просто запускаем симуляцию
    и выводим статистику хитрых метрик =)
    '''
    success, completed_requests, accelerators = simulate(N, df_sample.copy())

    total_reqs = len(completed_requests)
    if total_reqs == 0:
        print("Симуляция не вернула ни одного запроса.")
        return

    failed_reqs = [req for req in completed_requests if req.limit_failed]
    failed_count = len(failed_reqs)
    fail_rate = (failed_count / total_reqs) * 100

    print(f"\n{'=' * 50}")
    print(f"ЗАПУСК СИМУЛЯЦИИ: N (Кол-во GPU) = {N}")
    print(f"ПЕРВЫЕ M логов: = {M}")
    print(f"MAX_CHUNK_SIZE: = {MAX_CHUNK_SIZE}")
    print(f"MAX_BATCH_SIZE = {MAX_BATCH_SIZE} сколько макс рекв в батч")
    print(f"{'=' * 50}")

    print("\n--- БАЗОВАЯ СТАТИСТИКА ---")
    print(f"Всего запросов обработано: {total_reqs}")
    print(f"Провалено по SLA: {failed_count} ({fail_rate:.2f}%)")

    # Собираем сырые данные для перцентилей
    ttft_times = [
        (req.ttft_time - req.time_to_come)
        for req in completed_requests
        if req.ttft_time is not None
    ]
    decode_times = [
        (req.time_stage_3 / req.token_generation)
        for req in completed_requests
        if req.finish_time is not None and req.token_generation > 0
    ]

    if ttft_times:
        print("\n--- АНАЛИТИКА TTFT (Time To First Token) ---")
        print(f"Лимит SLA (P): {LIM_TTFT} сек")
        print(f"Среднее: {np.mean(ttft_times):.4f} сек")
        print(f"Медиана (P50): {np.percentile(ttft_times, 50):.4f} сек")
        print(f"P90: {np.percentile(ttft_times, 90):.4f} сек")
        print(f"P99: {np.percentile(ttft_times, 99):.4f} сек")
        print(f"Максимум: {np.max(ttft_times):.4f} сек")

    if decode_times:
        print("\n--- АНАЛИТИКА DECODE (Время на 1 токен генерации) ---")
        print(f"Лимит SLA (D): {LIM_GEN_NEXT_TOK} сек")
        print(f"Среднее: {np.mean(decode_times):.4f} сек")
        print(f"Медиана (P50): {np.percentile(decode_times, 50):.4f} сек")
        print(f"P90: {np.percentile(decode_times, 90):.4f} сек")
        print(f"P99: {np.percentile(decode_times, 99):.4f} сек")
        print(f"Максимум: {np.max(decode_times):.4f} сек")

    actual_makespan = max(req.finish_time for req in completed_requests if req.finish_time is not None)

    print("\n--- УТИЛИЗАЦИЯ И ПРОСТОЙ GPU (EMPIRICAL IDLE TIME) ---")
    total_idle_cluster = 0
    total_compute_cluster = 0

    for acc in accelerators:
        compute_time = acc.total_compute_time
        idle_time = actual_makespan - compute_time
        utilization = (compute_time / actual_makespan) * 100

        total_idle_cluster += idle_time
        total_compute_cluster += compute_time
        # Можно раскомментировать, чтобы посмотреть каждую карту отдельно:
        print(f"GPU {acc.gpu_id}: Утилизация {utilization:.1f}% | Работа: {compute_time:.1f}s | Простой: {idle_time:.1f}s")

    avg_cluster_utilization = (total_compute_cluster / (actual_makespan * N)) * 100
    print(f"Средняя загрузка кластера (Useful Compute): {avg_cluster_utilization:.2f}%")
    total_cluster_capacity = actual_makespan * N

    # Выводим корректное соотношение:
    print(f"Суммарный простой кластера (Idle Time): {total_idle_cluster:,.2f} сек из общих {total_cluster_capacity:,.2f} машино-секунд")
        # Базовая логика рекомендаций
    print("\n--- РЕКОМЕНДАЦИИ  --")
    if fail_rate == 0:
        print("Система работает идеально. Ресурсов достаточно.")
    else:
        # Пытаемся понять, где узкое горлышко
        ttft_fails = sum(1 for t in ttft_times if t > LIM_TTFT)
        decode_fails = sum(1 for d in decode_times if d > LIM_GEN_NEXT_TOK)

        if ttft_fails > decode_fails:
            print(" Диагноз: Проблема на этапе Prefill (превышен TTFT).")
            print(
                " Рекомендация: Запросы слишком долго ждут в очереди. Необходимо увеличить количество GPU (N)."
            )
        else:
            print("Диагноз: Проблема на этапе Decode (долгая генерация).")
            print(
                " Рекомендация: Слишком большой батч на генерации, из-за чего падает скорость. Возможно, стоит уменьшить MAX_CHUNK_SIZE или увеличить вычислительную мощность (COST_C)."
            )

# пример рассчета. Оптимальное количество рассчитывается в файле min_gpu_for_azure_logs.ipynd
run_analytics(N=11, df_sample=df_)
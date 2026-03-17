import heapq
import math
from collections import deque

import pandas as pd

# gzip по расширению
df = pd.read_csv("trace.csv.gz", compression="gzip")


# Глобальные константы. взяли всего одну GPU,
# где N = 1, K = ??? не указали, но надо
MEM_M = 800000  # mb   1000 MB ~ 1GB

MEM_X = 50  # вес самой картинки 1024px+ + weights для самой модели детекцим
MEM_Y = 1  # мб на 1 токен контекста (входной текст)
MEM_Z = 1  # мб на 1 сгенерированный токен ( выходной текст)
# ТОКЕНЫ контекста и генерации разные вещи и хранятся в VRAM

# compute_costs на препроцессинг картинки, обработку токена, время генерации 1 токена
COST_A = 0.5  # на картинку
COST_B = 0.01  # обработка 1 токена контекста
COST_C = 0.05  # 1 сгенерированный токен

# limitations global contants
LIM_TTFT = 500  # P ttft not more P
LIM_GEN_NEXT_TOK = 500  # D limit time stage 3 0.05 это sec/token

MAX_CHUNK_SIZE = 512  # tokens per forward pass


class Request:
    """
    Поля: id, время_прибытия, кол_во_картинок, токены_контекста, токены_генерации
    То что содержится в классе:
    start_processing_time - время когда gpu только НАЧАЛА работать
    ttft_time - закончился препроцессинг картинок и контекста
    finish_time - время сгенерированного последнего токена, который должен <= T (по ТЗ)
    limit_failed - флаг нарушение одно из требований по тз
    time_stage_3 - время до последнего токена
    max_gen_tokens - сколько у самого жирного реквеста будет токенов для генерации в батче
    request берез из df[['Arrival_Sec', 'ContextTokens', 'NumImages', 'GeneratedTokens']]

    """

    def __init__(self, id, time_to_come, token_context, num_images, token_generation):
        self.id = id
        self.time_to_come = time_to_come
        self.num_images = num_images
        self.token_context = token_context
        self.token_generation = token_generation

        # данные симуляции
        self.start_processing_time = None
        self.ttft_time = None

        self.images_left = self.num_images
        self.context_tokens_left = self.token_context
        self.tokens_left_to_generate = (
            self.token_generation
        )  # сколько осталось генерить
        self.allocated_vram = 0  # сколько памяти запрос схавал

        self.finish_time = None
        self.limit_failed = False
        self.time_stage_3 = None

    def get_memory_per_request(self):
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
    статус (Свободен/Занят), доступная_память, вычислительная_мощность
    """

    def __init__(self, status, memory, computing):
        self.status = status
        self.free_memory = MEM_M
        self.computing = computing
        self.active_batch = []  # наша VRAM
        self.is_ticking = (
            False  # это флаг того, что если загрузится гпу, то она в работе
        )


class Event:
    """
    класс события
    Поля: время_события, тип_события
    data - весь экземляр класса Accelerator
    list_current_batch - лист всех нынешнего батча
    """

    def __init__(self, time_event, type_event, data, list_current_batch=None):
        self.time_event = time_event
        self.type_event = type_event
        self.data = data
        self.list_current_batch = list_current_batch

    # для сравнения событий через магическиий метода __less than__
    def __lt__(self, other):
        return self.time_event < other.time_event


def make_heap_from_df(df):

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
    for row in df[
        ["Arrival_Sec", "ContextTokens", "NumImages", "GeneratedTokens"]
    ].itertuples():
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


def check_limitations(req):
    """
    раньше аргументы req
    ttft_duration = req.ttft_time - req.time_to_come
    time_per_token = time_stage_3 / max_gen_tokens
    if ttft_duration > LIM_TTFT or time_per_token > LIM_GEN_NEXT_TOK:
      req.limit_failed = True
    """
    # проверка на TTFT -в ttft_time там время префилл + time_to_come +
    if (req.ttft_time - req.time_to_come) > LIM_TTFT:
        req.limit_failed = True

    total_stage_3_time = req.finish_time - req.ttft_time
    average_time_per_token = total_stage_3_time / req.token_generation
    # проверка на лимит D
    if average_time_per_token > LIM_GEN_NEXT_TOK:
        req.limit_failed = True


"""
15 march динамический KV-CACHE + CB + CHUNKED PREFILL
512 токенов это sweet spot для A100/H100 и не нагружает ядра гпу чтобы
другие пользователи не заметили разницы в задержке response
но можно и больше - это гиперпараметр
"""


def scheduler_step(
    time_now, deque_requests, accelerator, scheduler_event, completed_requests
):
    # очистка завершенных в батче
    finished_requests = []

    for req in accelerator.active_batch:
        # случай когда все догенерилось
        if req.tokens_left_to_generate == 0:
            req.finish_time = time_now
            req.time_stage_3 = req.finish_time - req.ttft_time
            accelerator.free_memory += req.allocated_vram
            check_limitations(req)
            finished_requests.append(req)

    # теперь надо  из активного батча выкинуть завершенные через множества
    # и с охранением порядка
    finished_set = set(finished_requests)
    completed_requests.extend(finished_requests)
    accelerator.active_batch = [
        req for req in accelerator.active_batch if req not in finished_set
    ]

    # Continious Batching in flight если есть что-то в очереди и слезет в память
    while deque_requests:
        next_req = deque_requests[0]
        tokens_for_first_chunk = min(MAX_CHUNK_SIZE, next_req.token_context)
        initial_mem_needed = (next_req.num_images * MEM_X) + (
            tokens_for_first_chunk * MEM_Y
        )

        # ПРЕДОХРАНИТЕЛЬ ОТ ОТРАВЛЕННЫХ ЗАПРОСОВ
        if initial_mem_needed > MEM_M:
            deque_requests.popleft()  # Выкидываем запрос, он физически не влезет
            continue

        if initial_mem_needed <= accelerator.free_memory:
            req = deque_requests.popleft()
            accelerator.active_batch.append(req)

            # СРАЗУ бронируем память и под картинки, И под первый чанк префилла!
            img_mem = req.images_left * MEM_X
            chunk_mem = tokens_for_first_chunk * MEM_Y

            total_reserved = img_mem + chunk_mem
            accelerator.free_memory -= total_reserved
            req.allocated_vram += total_reserved

            req.images_left = 0
            # И сразу отнимаем токены из контекста, так как мы их забронировали
            req.context_tokens_left -= tokens_for_first_chunk

        else:
            # не хватка памяти
            break

    # GPU ate batch
    if not accelerator.active_batch:
        accelerator.is_ticking = False
        return

    # вычисление следующего такта (tick)
    B_size = len(accelerator.active_batch)
    # те кто не прошел prefill
    # CHUNKED PREFILL
    newbies = [i for i in accelerator.active_batch if i.ttft_time is None]
    decoders = [k for k in accelerator.active_batch if k.ttft_time is not None]

    step_time = 0

    # prefilling newbies if they exist
    if newbies:
        current_chunk_tokens = 0

        for req in newbies:
            tokens_to_compute = min(
                req.context_tokens_left + MAX_CHUNK_SIZE, MAX_CHUNK_SIZE
            )
            current_chunk_tokens += tokens_to_compute

            if req.context_tokens_left > 0:
                next_chunk = min(req.context_tokens_left, MAX_CHUNK_SIZE)
                mem_needed = next_chunk * MEM_Y

                if accelerator.free_memory >= mem_needed:
                    req.context_tokens_left -= next_chunk
                    accelerator.free_memory -= mem_needed
                    req.allocated_vram += mem_needed

            if current_chunk_tokens >= MAX_CHUNK_SIZE:
                break

        processed_decoders = 0
        for req in decoders:
            while accelerator.free_memory < MEM_Z:
                victim_req = max(
                    accelerator.active_batch, key=lambda r: r.tokens_left_to_generate
                )

                if (
                    victim_req is req
                    or victim_req.tokens_left_to_generate <= req.tokens_left_to_generate
                ):
                    break

                # выселяем
                accelerator.free_memory += victim_req.allocated_vram
                victim_req.allocated_vram = 0
                victim_req.context_tokens_left = victim_req.token_context
                victim_req.ttft_time = None
                accelerator.active_batch.remove(victim_req)
                deque_requests.appendleft(victim_req)

            if accelerator.free_memory >= MEM_Z:
                req.tokens_left_to_generate -= 1
                accelerator.free_memory -= MEM_Z
                req.allocated_vram += MEM_Z
                processed_decoders += 1

        time_prefill = current_chunk_tokens * COST_B
        time_decode = processed_decoders * COST_C
        step_time = (time_prefill + time_decode) * math.sqrt(B_size)

        for req in newbies:
            if req.context_tokens_left == 0 and req.ttft_time is None:
                req.ttft_time = time_now + step_time
    else:
        # нету новеньких
        processed_decoders = 0
        for req in decoders:
            # Пытаемся освободить память, если ее нет
            # EVICTION POLICY
            while accelerator.free_memory < MEM_Z:
                victim_req = max(
                    accelerator.active_batch, key=lambda r: r.tokens_left_to_generate
                )

                if (
                    victim_req is req
                    or victim_req.tokens_left_to_generate <= req.tokens_left_to_generate
                ):
                    break

                accelerator.free_memory += victim_req.allocated_vram
                victim_req.allocated_vram = 0
                victim_req.context_tokens_left = victim_req.token_context
                victim_req.ttft_time = None

                accelerator.active_batch.remove(victim_req)
                deque_requests.appendleft(victim_req)

            # Если память есть (или мы ее освободили) - генерируем
            if accelerator.free_memory >= MEM_Z:
                req.tokens_left_to_generate -= 1
                accelerator.free_memory -= MEM_Z
                req.allocated_vram += MEM_Z
                processed_decoders += 1

        # время чистого декода
        step_time = (processed_decoders * COST_C) * math.sqrt(B_size)

        # создаем событие
        # типо трансформер прошёл один forward pass
        this_event = Event(
            time_event=time_now + step_time, type_event="TICK_DONE", data=accelerator
        )
        heapq.heappush(scheduler_event, this_event)
        accelerator.is_ticking = True


def simulate(N, df):
    """
    df - наши логи
    N - количество гпу
    """
    time_now = 0
    list_accelerators = [Accelerator("FREE ACCELERATOR", MEM_M, 1) for i in range(N)]

    deque_requests = deque()
    completed_requests = []  # успешно завершенные запросы

    # взяли и внесли в кучу запросы
    scheduler_event = make_heap_from_df(df)

    while scheduler_event:
        this_event = heapq.heappop(scheduler_event)
        time_now = this_event.time_event

        if this_event.type_event == "NEW REQUEST":
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
                    scheduler_step(
                        time_now,
                        deque_requests,
                        acc,
                        scheduler_event,
                        completed_requests,
                    )

    return True, completed_requests


find_necessary_N = 0
success_requests = []

M = 500
df_ = df.head(M)
print(f" M = {M}")
for i in range(1, 100):
    print(f"симулируем для количества карточек N = {i}")
    success, current_completed_requests = simulate(
        i, df_.copy()
    )  # берем результат из функции

    flag_sla = False
    for req in current_completed_requests:  # проверяем именно этот запуск
        if req.limit_failed:
            flag_sla = True
            print("Noooo, ne poluchilos")
            break

    if not flag_sla and len(current_completed_requests) > 0:  # ПРЕДОХРАНИТЕЛЬ
        find_necessary_N = i
        success_requests = current_completed_requests
        print(f"Победа! Минимальное N = {i}")
        break  # обязательно выходим, мы нашли минимум

"""Теперь сбор статистики с симуляции, где мощей гпу хватило"""

list_sum_all_time_stages = []  # НАБОР T
list_samples_ttft = []  # НАБОР TTFT ДЛЯ КАЖДОГО ЛОГА
for req in success_requests:
    list_sum_all_time_stages.append(req.finish_time - req.time_to_come)
    list_samples_ttft.append(req.ttft_time - req.time_to_come)

print(list_samples_ttft)
print(list_sum_all_time_stages)

import statistics

if find_necessary_N > 0:
    print("\n--- TTFT (Time To First Token) ---")
    print(f"Минимум: {min(list_samples_ttft):.2f} сек")
    print(f"Максимум: {max(list_samples_ttft):.2f} сек")
    print(f"Среднее: {statistics.mean(list_samples_ttft):.2f} сек")
    print(f"Медиана: {statistics.median(list_samples_ttft):.2f} сек")

    print("\n--- TOTAL TIME FROM REQUEST TO FULL RESPONSE ---")
    print(f"Минимум: {min(list_sum_all_time_stages):.2f} сек")
    print(f"Максимум: {max(list_sum_all_time_stages):.2f} сек")
    print(f"Среднее: {statistics.mean(list_sum_all_time_stages):.2f} сек")
    print(f"Медиана: {statistics.median(list_sum_all_time_stages):.2f} сек")

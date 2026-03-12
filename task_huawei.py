import pandas as pd
from collections import deque
import heapq
import math

# gzip по расширению
df = pd.read_csv('trace.csv.gz', compression='gzip')

# Глобальные константы. взяли всего одну GPU,
# где N = 1, K = ??? не указали, но надо
MEM_M = 80000 # 80 gb memory

# для теста взяли оч маленький вес и большие рамки SLA 
MEM_X = 5 # вес самой картинки 1024px+ + weights для самой модели детекцим
MEM_Y  = 1 # мб на 1 токен контекста (входной текст)
MEM_Z = 1 # мб на 1 сгенерированный токен ( выходной текст)
# ТОКЕНЫ контекста и генерации разные вещи и хранятся в VRAM

#compute_costs на препроцессинг картинки, обработку токена, время генерации 1 токена
COST_A = 0.5 # на картинку
COST_B = 0.01 # обработка 1 токена контекста
COST_C = 0.05 # 1 сгенерированный токен

# limitations global contants
LIM_TTFT = 5000.0 # P ttft not more P
LIM_GEN_NEXT_TOK = 10  # D limit time stage 3 0.05 это sec/token

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
    self.finish_time = None
    self.limit_failed = False
    self.time_stage_3 = None
    self.max_gen_tokens = None

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
  df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

  # Берем время самого первого запроса
  start_time = df['TIMESTAMP'].min()

  # Считаем разницу в секундах (с долями миллисекунд)
  df['Arrival_Sec'] = (df['TIMESTAMP'] - start_time).dt.total_seconds()
  # если не отсортированно изначально
  df.sort_values(by='TIMESTAMP', inplace=True)

  # из дата фрейма в инициализацию календаря
  for row in df[['Arrival_Sec', 'ContextTokens', 'NumImages', 'GeneratedTokens']].itertuples():
    req = Request(row.Index, row.Arrival_Sec, row.ContextTokens, row.NumImages, row.GeneratedTokens)

    event = Event(req.time_to_come, "NEW REQUEST", req)
    scheduler_event.append(event)

  # подаем лист, делает все на месте, лист теперь куча
  heapq.heapify(scheduler_event)
  return scheduler_event


def give_job(deque_requests, list_accelerators, time_now, scheduler_event):

  for accelerator in list_accelerators:
    if accelerator.status == "FREE ACCELERATOR" and deque_requests:

      # напихать жадным образом

      current_batch = [] # собираем пачку реквестов


      while deque_requests:
        req = deque_requests[0] # только смотрим, без извлечения

        if req.get_memory_per_request() > MEM_M:
          print("Error, this request is bigger than all VRAM --> Rejection")
          print(f"LOGS: {req}")
          deque_requests.popleft()
          continue
        if req.get_memory_per_request() <= accelerator.free_memory:
          req = deque_requests.popleft() # реальное извлечение
          current_batch.append(req)
          accelerator.free_memory -= req.get_memory_per_request()
        else:
          print("Not enough memory for batching anymore, batch go to computing")
          break

      if current_batch: # если накопилось хоть чето
        accelerator.status = 'BUSY ACCELERATOR'
        B_size = len(current_batch)

        # памяти мы занимаем линейно, вычисления квадратично по ощущениям

        sum_images = sum([req.num_images for req in current_batch])
        sum_context = sum([req.token_context for req in current_batch])

        # или типо параллелим и вот мы ждем когда жирный батч досчитается, тогда и весь батч посчитан
        # типо паддинг или бабблинг
        #!!!! можно ли в батч собирать хитрее? которые плюч минус одинаково большие + большие

        # поэтому тут надо Continious batching присобачить
        max_gen_tokens = max([req.token_generation for req in current_batch])

        # эвристика квадратичного корня вычисления
        speed_factor_computing = math.sqrt(B_size)

        # вычисляем время
        time_stage_1 = (sum_images * COST_A) * speed_factor_computing
        time_stage_2 = (sum_context * COST_B) * speed_factor_computing
        time_stage_3 = (max_gen_tokens * COST_C) * speed_factor_computing


        total_work_time = time_stage_1 + time_stage_2 + time_stage_3
        job_finish_time = time_now + total_work_time


        # обновим состояния для всей пачки запросов
        for req in current_batch:
          req.start_processing_time = time_now
          req.ttft_time = time_now + time_stage_1 + time_stage_2
          req.finish_time = job_finish_time
          req.time_stage_3 = time_stage_3
          req.max_gen_tokens = max_gen_tokens


        new_event = Event(job_finish_time, "FREE ACCELERATOR", accelerator, current_batch)
        heapq.heappush(scheduler_event, new_event)
        print(f'{time_now} Accelerator took batch {B_size} requests.Busy until {job_finish_time}')

def check_limitations(req, time_stage_3, max_gen_tokens):
  ttft_duration = req.ttft_time - req.time_to_come
  time_per_token = time_stage_3 / max_gen_tokens
  if ttft_duration > LIM_TTFT or time_per_token > LIM_GEN_NEXT_TOK:
    req.limit_failed = True

def simulate(N, df):
  '''
  df - наши логи
  N - количество гпу
  '''
  time = 0
  list_accelerators = [Accelerator("FREE ACCELERATOR", MEM_M, 1) for i in range(N)]

  deque_requests = deque()
  completed_requests = [] # успешно завершенные запросы

  # взяли и внесли в кучу запросы
  scheduler_event = make_heap_from_df(df)

  # main cycle loop

  while scheduler_event:
    # heapop уже достает ссылку на ту видеокарту на которой создан event
    # ссылка сохраняется
    this_event = heapq.heappop(scheduler_event)
    time_now = this_event.time_event

    if this_event.type_event == "NEW REQUEST":

      deque_requests.append(this_event.data) # именно тут я помещаю чето в очередь
      give_job(deque_requests, list_accelerators, time_now, scheduler_event)


    if this_event.type_event == "FREE ACCELERATOR":
      # подсчитаем нарушения SLA
      for req in this_event.list_current_batch:
        check_limitations(req, req.time_stage_3, req.max_gen_tokens)
        if req.limit_failed == False:
          completed_requests.append(req)
        else:
          return False, []

      #достает ссылку на экзепляр ускорителя
      # к которой привязано определенное событие с определенным батчом
      this_accelerator = this_event.data
      # симулировали очистистку памяти
      this_accelerator.free_memory = MEM_M
      this_accelerator.status = "FREE ACCELERATOR"

      give_job(deque_requests, list_accelerators, time_now, scheduler_event)

  return True, completed_requests

find_necessary_N = 0
success_requests = []

M = 50
df_ = df.head(M)

for i in range(1, 10):

  print(f'симулируем для количества карточек N = {i}')
  success, requests = simulate(i, df_.copy())

  if success == True:
    find_necessary_N = i
    success_requests = requests
    print(f'минимальное число для вычислений первых {M} логов --> {i} GPU')
    break

"""
Теперь сбор статистики с симуляции, где мощей гпу хватило
где очень щадящие требования
MEM_M = 80000 # 80 gb memory
LIM_TTFT = 5000.0 # P ttft not more P
LIM_GEN_NEXT_TOK = 10  # D limit time stage 3 0.05 это sec/token
"""

list_sum_all_time_stages = [] # НАБОР T
list_samples_ttft = [] # НАБОР TTFT ДЛЯ КАЖДОГО ЛОГА
for req in success_requests:
  list_sum_all_time_stages.append(req.finish_time  - req.time_to_come)
  list_samples_ttft.append(req.ttft_time - req.time_to_come)

import statistics

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


'''
по итогу с параметрами выше потребовалось 2 GPU 
также не введена константа K - computing

--- TTFT (Time To First Token) ---
Минимум: 7.70 сек
Максимум: 3787.99 сек
Среднее: 2830.46 сек
Медиана: 3778.14 сек

--- TOTAL TIME FROM REQUEST TO FULL RESPONSE ---
Минимум: 16.29 сек
Максимум: 4024.67 сек
Среднее: 3017.73 сек
Медиана: 4014.81 сек

'''






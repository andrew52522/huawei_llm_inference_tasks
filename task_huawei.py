
"""#  сама симуляция на сырых логах!!"""

# !wget -O trace.csv.gz https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureLMMInferenceTrace_multimodal.csv.gz

import pandas as pd
from collections import deque
import heapq
import math

# gzip по расширению
df = pd.read_csv('trace.csv.gz', compression='gzip')
df = df.head(50)


# Глобальные константы. взяли всего одну GPU,
# где N = 1, K = ??? не указали, но надо
MEM_M = 80000 # 80 gb memory

MEM_X = 500 # вес самой картинки 1024px+ + weights для самой модели детекцим
MEM_Y  = 1 # мб на 1 токен контекста (входной текст)
MEM_Z = 1 # мб на 1 сгенерированный токен ( выходной текст)
# ТОКЕНЫ контекста и генерации разные вещи и хранятся в VRAM

#compute_costs на препроцессинг картинки, обработку токена, время генерации 1 токена
COST_A = 0.5 # на картинку
COST_B = 0.01 # обработка 1 токена контекста
COST_C = 0.05 # 1 сгенерированный токен

class Request:
  """
  Поля: id, время_прибытия, кол_во_картинок, токены_контекста, токены_генерации
  request берез из df[['Arrival_Sec', 'ContextTokens', 'NumImages', 'GeneratedTokens']]
  """
  def __init__(self, id, time_to_come, token_context, num_images, token_generation):
    self.id = id
    self.time_to_come = time_to_come
    self.num_images = num_images
    self.token_context = token_context
    self.token_generation = token_generation

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
  data - весь экземляр класса Requests, Accelerator
  """
  def __init__(self, time_event, type_event, data):
    self.time_event = time_event
    self.type_event = type_event
    self.data = data

  # для сравнения событий через магическиий метода __less than__
  def __lt__(self, other):
    return self.time_event < other.time_event

scheduler_event = []
deque_requests = deque()
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

"""1 экземпляр GPU"""

time = 0
list_accelerators = []
# computing пока не используем здесь
accelerator = Accelerator("FREE ACCELERATOR", MEM_M, 1)
list_accelerators.append(accelerator)

def give_job(deque_requests, list_accelerators, time_now, scheduler_event):

  for accelerator in list_accelerators:
    if accelerator.status == "FREE ACCELERATOR" and deque_requests:

     
      # напихать жадным образом

      current_batch = [] # собираем пачку


      while deque_requests:
        req = deque_requests[0]

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

        # памяти мы занимаем линейно, вычисления квадратично как дали эвристику

        sum_images = sum([req.num_images for req in current_batch])
        sum_context = sum([req.token_context for req in current_batch])

        # типо параллелим и вот мы ждем когда жирный батч досчитается, тогда и весь батч посчитан
        # поэтому берем max()
        # типо паддинг или бабблинг проблема
        #!!!! можно ли в батч собирать хитрее? которые плюч минус одинаково большие + большие

        # поэтому тут надо Continious batching присобачить
        max_gen_tokens = max([req.token_generation for req in current_batch])

        # эвристика
        speed_factor_computing = math.sqrt(B_size)

        # вычисляем время
        time_stage_1 = (sum_images * COST_A) * speed_factor_computing
        time_stage_2 = (sum_context * COST_B) * speed_factor_computing
        time_stage_3 = (max_gen_tokens * COST_C) * speed_factor_computing


        total_work_time = time_stage_1 + time_stage_2 + time_stage_3
        job_finish_time = time_now + total_work_time

        new_event = Event(job_finish_time, "FREE ACCELERATOR", accelerator)
        heapq.heappush(scheduler_event, new_event)
        print(f'{time_now} Accelerator took batch {B_size} requests.Busy until {job_finish_time}')



# main cycle loop

while scheduler_event:
  # heapop уже достает ссылку на ту видеокарту на которой создан event
  # ссылка сохраняется
  this_event = heapq.heappop(scheduler_event)
  time_now = this_event.time_event

  if this_event.type_event == "NEW REQUEST":

    deque_requests.append(this_event.data)
    give_job(deque_requests, list_accelerators, time_now, scheduler_event)

  if this_event.type_event == "FREE ACCELERATOR":
     #достает ссылку на экзепляр ускорителя
     # к которой привязано определенное событие с определенным батчом
    this_accelerator = this_event.data
    # симулировали очистистку памяти
    this_accelerator.free_memory = MEM_M
    this_accelerator.status = "FREE ACCELERATOR"

    give_job(deque_requests, list_accelerators, time_now, scheduler_event)


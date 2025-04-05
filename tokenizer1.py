import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from datasets import load_dataset, interleave_datasets
from transformers import Trainer, TrainingArguments
from time import sleep
import json

# Пути к моделям
BASE_MODEL_PATH = r"D:/Unlocked_Model"  # Папка с локальной моделью
LORA_ADAPTER_PATH = r"D:/MyFreedomGPT-LoRA".replace("\\", "/")  # Исправляем путь для LoRA адаптера
MERGED_MODEL_PATH = r"D:/MyFreedomGPT-Merged"  # Папка для сохранения объединённой модели

# Загружаем только структуру модели (конфигурацию)
print("🔄 Загружаем конфигурацию модели (структура)...")
base_model = None
for attempt in range(3):  # Попробуем несколько раз
    try:
        # Загружаем только конфигурацию модели без весов
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH)  # Загружаем конфигурацию модели
        base_model = AutoModelForCausalLM.from_config(config)  # Загружаем модель без весов
        print("✅ Модель успешно загружена.")
        break  # Выход из цикла при успешной загрузке
    except RuntimeError as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        if attempt < 2:
            print("🔄 Повторная попытка через 5 секунд...")
            sleep(5)
        else:
            print("⚠️ Все попытки загрузки модели не увенчались успехом. Переход к следующему шагу.")

# Если модель не загружена, выводим сообщение и завершаем выполнение
if base_model is None:
    print("❌ Модель не загружена. Скрипт завершится.")
    exit(1)

# Используем CPU для вычислений
device = "cpu"
print("📍 Используем CPU для вычислений")
base_model.to(device)  # Переводим модель на CPU

# 4️⃣ **Загружаем датасеты**
print("📥 Загружаем датасеты...")

dataset_urls = {
    "code": "bigcode/the-stack",
    "news": "cc_news",
    "pile": "EleutherAI/pile",
    "openwebtext": "openwebtext",  # Оставляем OpenWebText
    "daily_dialog": "daily_dialog"  # Добавляем DailyDialog для более естественных диалогов
}

datasets = []
for name, url in dataset_urls.items():
    try:
        print(f"📂 Загружаем {name} (режим стриминга)...")
        ds = load_dataset(url, split="train", streaming=True, trust_remote_code=True)
        ds = ds.take(50)  # 🔥 Ограничиваем до 50 примеров (уменьшаем память)
        datasets.append(ds)
        print(f"✅ {name} успешно загружен.")
    except Exception as e:
        print(f"❌ Ошибка загрузки {name}: {e}")

# ✅ **Объединяем датасеты**
if datasets:
    dataset = interleave_datasets(datasets, stopping_strategy="first_exhausted")
    print(f"✅ Датасеты загружены.")
else:
    dataset = None
    print("⚠️ Не удалось загрузить ни один датасет.")

# 5️⃣ **Настроим LoRA для обучения**
print("🔧 Настроим LoRA для обучения...")

# Проверяем, существует ли файл конфигурации для LoRA адаптера
adapter_config_path = os.path.join(LORA_ADAPTER_PATH, 'adapter_config.json')
if not os.path.exists(adapter_config_path):
    print("❌ Файл конфигурации LoRA адаптера не найден. Создаём новый...")

    # Создаём минимальную конфигурацию для LoRA адаптера
    adapter_config = {
        "adapter_type": "LoRA",  # Тип адаптера
        "base_model": BASE_MODEL_PATH,  # Базовая модель
        "r": 8,  # Размер скрытого слоя для LoRA (можно изменить в зависимости от модели)
        "lora_alpha": 16,  # Коэффициент LoRA (можно настроить)
        "lora_dropout": 0.1,  # Dropout для LoRA (можно настроить)
        "trainable": True,  # Все параметры LoRA будут обучаемыми
    }

    # Сохраняем конфигурацию
    os.makedirs(LORA_ADAPTER_PATH, exist_ok=True)
    with open(adapter_config_path, 'w') as f:
        json.dump(adapter_config, f, indent=4)
    print(f"✅ Файл конфигурации LoRA адаптера создан по пути: {adapter_config_path}")

# Загружаем LoRA адаптер
try:
    print(f"🔧 Загружаем LoRA адаптер из {LORA_ADAPTER_PATH}...")
    lora_adapter = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    print("✅ LoRA адаптер успешно загружен.")
except Exception as e:
    print(f"❌ Ошибка при загрузке LoRA адаптера: {e}")
    exit(1)

# 6️⃣ **Обучение LoRA адаптера**
print("📚 Начинаем обучение LoRA адаптера...")

# Загружаем токенизатор (обязательно)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Используем tokenization и форматирование данных
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Применяем препроцессинг
train_dataset = dataset.map(preprocess_function, batched=True)

# Настройка параметров для обучения
training_args = TrainingArguments(
    output_dir="./results",          # Папка для хранения результатов
    evaluation_strategy="epoch",     # Оценка модели после каждого эпока
    learning_rate=2e-5,              # Скорость обучения
    per_device_train_batch_size=4,   # Размер батча уменьшен для снижения нагрузки на память
    gradient_accumulation_steps=4,   # Градиентное накопление (имитация большего батча)
    num_train_epochs=3,              # Количество эпох
    weight_decay=0.01,               # Регуляризация
    logging_dir="./logs",            # Папка для логов
    fp16=True,                       # Использование FP16 для ускорения обучения и экономии памяти
    load_best_model_at_end=True,     # Загружать лучшую модель в конце обучения
)

# Обучающий процесс
trainer = Trainer(
    model=lora_adapter,  # Используем LoRA адаптер
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Обучение LoRA адаптера
trainer.train()

# 7️⃣ **Слияние LoRA адаптера с моделью**
print("🔄 Сливаем LoRA адаптер с моделью...")
merged_model = lora_adapter.merge_and_unload()  # Слияние адаптера с моделью

# 8️⃣ **Сохранение объединённой модели**
if not os.path.exists(MERGED_MODEL_PATH):
    os.makedirs(MERGED_MODEL_PATH)  # Создаём папку для сохранения модели

print("💾 Сохраняем объединённую модель...")
merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)  # Сохраняем токенизатор
print(f"✅ Объединённая модель с LoRA адаптером сохранена в {MERGED_MODEL_PATH}")

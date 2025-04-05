import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from datasets import load_dataset, interleave_datasets
from transformers import Trainer, TrainingArguments
from time import sleep
import shutil
from google.colab import files
import json

# Указываем ссылку на репозиторий модели на Hugging Face
BASE_MODEL_REPO_URL = "https://huggingface.co/MisterHolY/Unloked_Model-Mistral.7-B"  # Ссылка на твою модель Hugging Face
MERGED_MODEL_PATH = "/content/merged_model"  # Путь для сохранения объединённой модели

# Загружаем модель из Hugging Face
print("🔄 Загружаем модель из репозитория Hugging Face...")
base_model = None
for attempt in range(3):  # Попробуем несколько раз
    try:
        # Загружаем только конфигурацию модели
        config = AutoConfig.from_pretrained(BASE_MODEL_REPO_URL)  # Загружаем конфигурацию модели
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO_URL, config=config)  # Загружаем модель с весами
        print("✅ Модель успешно загружена.")
        break
    except RuntimeError as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        if attempt < 2:
            print("🔄 Повторная попытка через 5 секунд...")
            sleep(5)
        else:
            print("⚠️ Все попытки загрузки модели не увенчались успехом. Переход к следующему шагу.")

# Если модель не загружена, завершаем выполнение
if base_model is None:
    print("❌ Модель не загружена. Скрипт завершится.")
    exit(1)

# Используем CPU для вычислений
device = "cpu"
base_model.to(device)

# Загружаем датасеты
print("📥 Загружаем датасеты...")
dataset_urls = {
    "code": "bigcode/the-stack",
    "news": "cc_news",
    "pile": "EleutherAI/pile",
    "openwebtext": "openwebtext",
    "daily_dialog": "daily_dialog"
}
datasets = []
for name, url in dataset_urls.items():
    try:
        print(f"📂 Загружаем {name} (режим стриминга)...")
        ds = load_dataset(url, split="train", streaming=True, trust_remote_code=True)
        ds = ds.take(50)  # Ограничиваем до 50 примеров
        datasets.append(ds)
        print(f"✅ {name} успешно загружен.")
    except Exception as e:
        print(f"❌ Ошибка загрузки {name}: {e}")

if datasets:
    dataset = interleave_datasets(datasets, stopping_strategy="first_exhausted")
    print("✅ Датасеты загружены.")
else:
    dataset = None
    print("⚠️ Не удалось загрузить ни один датасет.")

# Настроим LoRA для обучения
print("🔧 Настроим LoRA для обучения...")
LORA_ADAPTER_PATH = "/content/lora_adapter"  # Путь для LoRA адаптера

# Проверка и создание конфигурации LoRA адаптера
adapter_config_path = os.path.join(LORA_ADAPTER_PATH, 'adapter_config.json')
if not os.path.exists(adapter_config_path):
    adapter_config = {
        "adapter_type": "LoRA",
        "base_model": BASE_MODEL_REPO_URL,  # Путь к базовой модели
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "trainable": True,
    }
    os.makedirs(LORA_ADAPTER_PATH, exist_ok=True)
    with open(adapter_config_path, 'w') as f:
        json.dump(adapter_config, f, indent=4)

# Загружаем LoRA адаптер
try:
    print("🔧 Загружаем LoRA адаптер...")
    lora_adapter = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    print("✅ LoRA адаптер успешно загружен.")
except Exception as e:
    print(f"❌ Ошибка при загрузке LoRA адаптера: {e}")
    exit(1)

# Обучение LoRA адаптера
print("📚 Начинаем обучение LoRA адаптера...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO_URL)
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
train_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results", 
    evaluation_strategy="epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, 
    num_train_epochs=3,
    weight_decay=0.01, 
    logging_dir="./logs",
    fp16=True,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=lora_adapter, 
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Слияние LoRA адаптера с моделью
print("🔄 Сливаем LoRA адаптер с моделью...")
merged_model = lora_adapter.merge_and_unload()

# Сохраняем объединённую модель
print("💾 Сохраняем объединённую модель...")
if not os.path.exists(MERGED_MODEL_PATH):
    os.makedirs(MERGED_MODEL_PATH)

merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

# Создаём архив
shutil.make_archive(MERGED_MODEL_PATH, 'zip', MERGED_MODEL_PATH)

# Скачиваем архив
print("📥 Скачиваем архив модели...")
files.download(f'{MERGED_MODEL_PATH}.zip')



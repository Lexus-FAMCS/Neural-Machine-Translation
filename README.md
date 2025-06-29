# Neural Machine Translation 🇬🇧➡️🇷🇺

Переводчик с английского на русский на основе **Encoder-Decoder Transformer**.  
Написан на PyTorch. Вдохновлён оригинальной статьёй ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).

---

## 📚 Описание

Этот репозиторий содержит чистую реализацию трансформера для задачи машинного перевода.  
Модель обучена на паре языков **английский ↔ русский**. Использует собственный токенайзер или Hugging Face Tokenizers.

- ✨ **Архитектура**: Encoder-Decoder Transformer  
- 🔑 **Функции**:
  - Позиционное кодирование
  - Masked Multi-Head Attention
  - Cross-Attention
  - Feedforward Network
  - Авто-регрессивная генерация перевода
- 🗃️ **Данные**: ты можешь подключить свой датасет пар предложений (например, [ManyThings](https://www.manythings.org/anki/)) или любой другой параллельный корпус.

---

## 🚀 Быстрый старт

### 1️⃣ Клонируй репозиторий
```bash
git clone https://github.com/yourusername/translation-transformer.git
cd translation-transformer

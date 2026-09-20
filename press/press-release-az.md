# Press-reliz (AZ) — 2026-09-20-dən sonra dərc üçün

> Kvadrat mötərizədəki yerlər səhər 10:00-da yekun rəqəmlərlə doldurulacaq.

**Bakıda hazırlanan açıq model süni intellektin "uydurduğu" cavabları 30 dildə tapır**

Bakı, [tarix] — Azerbaycan şirkəti SIBA ("Süni İntellekt Biznes Avtomatlaşdırma" MMC) süni intellekt köməkçilərinin sənədlərdə olmayan məlumatı — səhv qiymət, saat, ünvan, telefon — necə "uydurduğunu" tapan açıq mənbəli aləti və ilk çoxdilli reytinq cədvəlini təqdim edir. Layihə **grounded.siba.az** ünvanında hər kəs üçün açıqdır: istifadəçi köməkçinin cavabını və istinad etdiyi sənədi yapışdırır, dəstəklənməyən iddialar qırmızı ilə işarələnir.

**Problem.** Bu gün dünyada ən çox yüklənən açıq halüsinasiya detektorları — Vectara HHEM-2.1 və LettuceDetect — yalnız ingilis dili üçün öyrədilib. SIBA-nın 11 dildə (Azərbaycan, rus, türk, ukrayna, qazax, ərəb, fars, hind, indoneziya, vyetnam və ingilis) 154 halda apardığı ölçmə göstərir ki, HHEM ingilis olmayan istənilən mətni — düzgün olsa belə — halüsinasiya kimi işarələyir: rus, ərəb, fars və indoneziya dillərində 22 düzgün cavabın hamısını "səhv" sayır. Yəni alət dili tanıyır, iddianı yox.

**Həll.** SIBA iki şey təqdim edir:
1. `groundedness` — sıfır asılılığı olan Python paketi (`pip install groundedness`, MIT lisenziyası): istənilən açıq dil modelini hakim kimi işlədir və cavabın öz dilində dəstəklənməyən iddiaları sadalayır, sonra cavabı onlarsız yenidən yazır. Testdə Groq üzərində işləyən açıq modellər 132 planlaşdırılmış səhvin [132]-sini sıfır yalançı siqnalla tapıb; 27 milyard parametrli model bunu orta hesabla 0,36 saniyəyə edir.
2. **SIBA-nın öz detektoru** — 30 dildə [~12 000] etiketlənmiş cavab üzərində öyrədilmiş çoxdilli model. Bulud və API tələb etmir: Bakıda bir Raspberry Pi üzərində işləyir və grounded.siba.az-da limitsiz, pulsuz sınaqdadır. 154 hallıq testdə [X/132] səhvi tapır, [Y/22] yalançı siqnal verir.

**Niyə vacibdir.** Sənədlərdən cavab verən çat-botlar (mağaza, klinika, bank, dövlət xidməti) ən çox məhz bu cür səhv edir — və Azərbaycan dili üçün indiyədək bunu yoxlayan heç bir alət olmayıb. SIBA eyni yoxlamanı öz müştəri köməkçisinin bütün kanallarında — sayt, WhatsApp, Telegram, telefon — hər cavaba tətbiq edir.

**Nəticələr açıqdır.** Bütün test halları, xam model cavabları, öyrətmə skriptləri və texniki hesabat GitHub-da: github.com/aghasalim/groundedness. Yeni dil əlavə etmək bir JSON qeydidir; müəlliflər digər dillərin daşıyıcılarını halları yoxlamağa dəvət edir.

**Müəlliflər:** Ağasəlim Mustafazadə (həmtəsisçi, süni intellekt üzrə rəhbər) və Teymur Eyvazov (həmtəsisçi, əməliyyatlar üzrə rəhbər), SIBA, Bakı.

Əlaqə: salim@siba.az · siba.az · grounded.siba.az

---

## Qısa versiya — Telegram / LinkedIn / icma qrupları üçün

Süni intellekt köməkçisi sənəddə olmayan qiyməti "uydurur" — bunu Azərbaycan dilində tutan alət indiyədək yox idi. 🇦🇿

Biz SIBA-da açıq mənbəli **groundedness** paketini və 30 dildə öyrədilmiş öz detektorumuzu buraxdıq. Detektor Bakıda bir Raspberry Pi-də işləyir, pulsuz və limitsizdir: cavabı və sənədi yapışdırın, uydurma iddialar qırmızı yanır → **grounded.siba.az**

Maraqlı tapıntı: dünyada ən çox yüklənən açıq halüsinasiya detektoru (Vectara HHEM) ingilis olmayan hər cavabı — düzgün olsa belə — halüsinasiya sayır. 11 dildə 154 halla yoxladıq; rus, ərəb, fars, indoneziya dillərində 22 düzgün cavabın 22-sini "səhv" adlandırır.

Kod, data, texniki hesabat: github.com/aghasalim/groundedness · `pip install groundedness`

#süniintellekt #AI #Azerbaijan #NLP #hallucination

---

## Hara göndərmək

Media (xəbər göndərmə forması / redaksiya ünvanı saytlarında):
- Report.az — "Texnologiya" bölməsi
- APA.az — İKT
- Trend.az — İKT
- Tech.az və ICTnews.az — sahə saytları, belə xəbəri birbaşa götürürlər
- Oxu.az / Qafqazinfo — geniş auditoriya
- Azərbaycan Respublikası Rəqəmsal İnkişaf və Nəqliyyat Nazirliyi / İnnovasiya və Rəqəmsal İnkişaf Agentliyi (IDDA) — "yerli məhsul" xəbəri kimi paylaşırlar
- Startup Azerbaijan (SABAH.lab, Innoland, Barama) — icma kanalları

İcma:
- Telegram: "AI Azerbaijan", "Data Science Azerbaijan", "Python Azerbaijan", ADA / BHOS / AzTU / UNEC tələbə qrupları
- LinkedIn: SIBA səhifəsi + şəxsi profil (heatmap şəkli ilə)
- Facebook: "Azerbaijan IT Community", "Proqramçılar"

Tövsiyə: əvvəl Tech.az / ICTnews (sürətli dərc), sonra Report/APA — onlar sahə saytlarında çıxan xəbəri daha asan götürür.

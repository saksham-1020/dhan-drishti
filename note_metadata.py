"""
note_metadata.py
=================
Static reference data: security-feature checklists, historical notes, and
the multilingual voice-announcement phrase book. Kept separate from app.py
so the UI logic file stays short and reviewable.
"""

NOTE_METADATA = {
    "background": {
        "denomination": 0, "title": "No Currency Detected", "series": "N/A",
        "color": "N/A", "dimensions": "N/A", "motif": "N/A", "year": "N/A",
        "features": [
            "Ensure the note is flat and centered.",
            "Avoid dark shadows or strong glare.",
            "Make sure fingers are not covering the main features of the note.",
        ],
        "funFact": "A dedicated background class prevents false positives on blank tables, hands, or other out-of-distribution clutter.",
        "accent": "#7c7295",
    },
    "ten_new": {
        "denomination": 10, "title": "Ten Rupee (New Note)", "series": "Mahatma Gandhi (New Series)",
        "color": "Chocolate Brown", "dimensions": "123 mm x 63 mm", "motif": "Sun Temple, Konark", "year": "2018",
        "features": [
            "See-through register with denominational numeral 10.",
            "Denominational numeral in Devanagari script.",
            "Portrait of Mahatma Gandhi at the center.",
            "Windowed security thread with 'Bharat' and 'RBI' inscriptions.",
        ],
        "funFact": "The Sun Temple motif represents Odisha's Konark heritage; its carved wheels double as sundials.",
        "accent": "#a0522d",
    },
    "ten_old": {
        "denomination": 10, "title": "Ten Rupee (Old Note)", "series": "Mahatma Gandhi Series",
        "color": "Orange-Brown / Pinkish", "dimensions": "137 mm x 63 mm", "motif": "Rhinoceros, Elephant, Tiger", "year": "2001",
        "features": [
            "Mahatma Gandhi portrait on the left obverse.",
            "RBI seal and Ashoka Pillar emblem on the right.",
            "Indian wildlife fauna motif on the reverse.",
        ],
        "funFact": "This series showcased India's ecological diversity through its wildlife back-panel art.",
        "accent": "#d2691e",
    },
    "twenty_new": {
        "denomination": 20, "title": "Twenty Rupee (New Note)", "series": "Mahatma Gandhi (New Series)",
        "color": "Greenish-Yellow", "dimensions": "129 mm x 63 mm", "motif": "Ellora Caves", "year": "2019",
        "features": [
            "See-through register with denominational numeral 20.",
            "Devanagari denominational numeral.",
            "Micro-lettering: RBI, Bharat, India, 20.",
        ],
        "funFact": "The Ellora Caves near Aurangabad are a UNESCO World Heritage rock-cut monastery complex.",
        "accent": "#8fbc8f",
    },
    "twenty_old": {
        "denomination": 20, "title": "Twenty Rupee (Old Note)", "series": "Mahatma Gandhi Series",
        "color": "Red-Orange", "dimensions": "147 mm x 63 mm", "motif": "Mount Harriet Lighthouse, Andaman", "year": "2001",
        "features": [
            "Mount Harriet lighthouse landscape, Andaman & Nicobar Islands.",
            "Ashoka Pillar and Mahatma Gandhi watermarks.",
            "Dual-tone reddish color scheme.",
        ],
        "funFact": "The reverse depicts North Bay, Port Blair, framed by coconut palms.",
        "accent": "#e9967a",
    },
    "fifty_new": {
        "denomination": 50, "title": "Fifty Rupee (New Note)", "series": "Mahatma Gandhi (New Series)",
        "color": "Fluorescent Blue", "dimensions": "135 mm x 66 mm", "motif": "Hampi Stone Chariot", "year": "2017",
        "features": [
            "Swachh Bharat logo and slogan on the reverse.",
            "Devanagari denominational numeral.",
            "Portrait of Mahatma Gandhi at the center.",
        ],
        "funFact": "Hampi, a UNESCO site in Karnataka, is home to the iconic stone chariot shrine.",
        "accent": "#00ced1",
    },
    "fifty_old": {
        "denomination": 50, "title": "Fifty Rupee (Old Note)", "series": "Mahatma Gandhi Series",
        "color": "Pink-Purple / Violet", "dimensions": "147 mm x 73 mm", "motif": "Parliament of India", "year": "1997",
        "features": [
            "Parliament House building on the reverse.",
            "Mahatma Gandhi portrait watermark.",
            "Intaglio-printed security thread.",
        ],
        "funFact": "The Parliament House motif symbolizes India's constitutional democracy.",
        "accent": "#ba55d3",
    },
    "hundred_new": {
        "denomination": 100, "title": "One Hundred Rupee (New Note)", "series": "Mahatma Gandhi (New Series)",
        "color": "Lavender", "dimensions": "142 mm x 66 mm", "motif": "Rani ki Vav stepwell, Patan", "year": "2018",
        "features": [
            "Rani ki Vav stepwell on the reverse.",
            "Color-shifting windowed security thread (green to blue on tilt).",
            "Intaglio raised-print Gandhi portrait.",
        ],
        "funFact": "Rani ki Vav is an 11th-century stepwell in Gujarat, built as an inverted temple to water.",
        "accent": "#7b68ee",
    },
    "hundred_old": {
        "denomination": 100, "title": "One Hundred Rupee (Old Note)", "series": "Mahatma Gandhi Series",
        "color": "Blue-Green", "dimensions": "157 mm x 73 mm", "motif": "Himalayan mountain range", "year": "1996",
        "features": [
            "Himalayan mountain landscape on the reverse.",
            "Latent image showing value '100' on the obverse.",
            "Ashoka Pillar emblem on the right.",
        ],
        "funFact": "The back panel features a view including Kanchenjunga, India's highest peak.",
        "accent": "#5f9ea0",
    },
    "two_hundred": {
        "denomination": 200, "title": "Two Hundred Rupee Note", "series": "Mahatma Gandhi (New Series)",
        "color": "Bright Yellow", "dimensions": "146 mm x 66 mm", "motif": "Sanchi Stupa", "year": "2017",
        "features": [
            "Sanchi Stupa Buddhist monument on the reverse.",
            "Intaglio-printed denominational numeral 200.",
            "Ashoka Pillar emblem on the right.",
        ],
        "funFact": "Sanchi Stupa, commissioned under Emperor Ashoka, is among India's oldest stone structures.",
        "accent": "#ffd700",
    },
    "five_hundred": {
        "denomination": 500, "title": "Five Hundred Rupee Note", "series": "Mahatma Gandhi (New Series)",
        "color": "Stone Grey", "dimensions": "150 mm x 66 mm", "motif": "Red Fort, Delhi", "year": "2016",
        "features": [
            "Red Fort with the Indian flag on the reverse.",
            "Intaglio-printed Mahatma Gandhi portrait.",
            "Devanagari denominational numeral.",
        ],
        "funFact": "The Red Fort served as the principal Mughal residence and is now a symbol of national independence.",
        "accent": "#808080",
    },
}

AUDIO_DICTIONARY = {
    "hi-IN": {
        "background": "कोई बैंक नोट नहीं मिला। कृपया नोट ठीक से दिखाएं।",
        "ten_new": "दस रुपये का नया नोट", "ten_old": "दस रुपये का पुराना नोट",
        "twenty_new": "बीस रुपये का नया नोट", "twenty_old": "बीस रुपये का पुराना नोट",
        "fifty_new": "पचास रुपये का नया नोट", "fifty_old": "पचास रुपये का पुराना नोट",
        "hundred_new": "सौ रुपये का नया नोट", "hundred_old": "सौ रुपये का पुराना नोट",
        "two_hundred": "दो सौ रुपये का नोट", "five_hundred": "पांच सौ रुपये का नोट",
        "warning_uncertain": "चेतावनी! मॉडल इस नोट को लेकर अनिश्चित है, कृपया दोबारा स्कैन करें।",
    },
    "en-IN": {
        "background": "No bank note detected. Please place a note in frame.",
        "ten_new": "Ten Rupee new note", "ten_old": "Ten Rupee old note",
        "twenty_new": "Twenty Rupee new note", "twenty_old": "Twenty Rupee old note",
        "fifty_new": "Fifty Rupee new note", "fifty_old": "Fifty Rupee old note",
        "hundred_new": "One hundred Rupee new note", "hundred_old": "One hundred Rupee old note",
        "two_hundred": "Two hundred Rupee note", "five_hundred": "Five hundred Rupee note",
        "warning_uncertain": "Warning! The model is uncertain about this note, please re-scan.",
    },
    "ta-IN": {
        "background": "நோட்டு கண்டறியப்படவில்லை. தயவுசெய்து நேராக வைக்கவும்.",
        "ten_new": "புதிய பத்து ரூபாய் நோட்டு", "ten_old": "பழைய பத்து ரூபாய் நோட்டு",
        "twenty_new": "புதிய இருபது ரூபாய் நோட்டு", "twenty_old": "பழைய இருபது ரூபாய் நோட்டு",
        "fifty_new": "புதிய ஐம்பது ரூபாய் நோட்டு", "fifty_old": "பழைய ஐம்பது ரூபாய் நோட்டு",
        "hundred_new": "புதிய நூறு ரூபாய் நோட்டு", "hundred_old": "பழைய நூறு ரூபாய் நோட்டு",
        "two_hundred": "இருநூறு ரூபாய் நோட்டு", "five_hundred": "ஐந்நூறு ரூபாய் நோட்டு",
        "warning_uncertain": "எச்சரிக்கை! மாதிரி நிச்சயமற்றது, மீண்டும் ஸ்கேன் செய்யவும்.",
    },
    "te-IN": {
        "background": "నోటు గుర్తించబడలేదు. దయచేసి స్పష్టంగా చూపించండి.",
        "ten_new": "కొత్త పది రూపాయల నోటు", "ten_old": "పాత పది రూపాయల నోటు",
        "twenty_new": "కొత్త ఇరవై రూపాయల నోటు", "twenty_old": "పాత ఇరవై రూపాయల నోటు",
        "fifty_new": "కొత్త యాభై రూపాయల నోటు", "fifty_old": "పాత యాభై రూపాయల నోటు",
        "hundred_new": "కొత్త వంద రూపాయల నోటు", "hundred_old": "పాత వంద రూపాయల నోటు",
        "two_hundred": "రెండు వందల రూపాయల నోటు", "five_hundred": "ఐదు వందల రూపాయల నోటు",
        "warning_uncertain": "హెచ్చరిక! మోడల్‌కు నిశ్చయత లేదు, దయచేసి మళ్ళీ స్కాన్ చేయండి.",
    },
    "bn-IN": {
        "background": "কোনো নোট শনাক্ত করা যায়নি। অনুগ্রহ করে স্পষ্টভাবে দেখান।",
        "ten_new": "নতুন দশ টাকার নোট", "ten_old": "পুরানো দশ টাকার নোট",
        "twenty_new": "নতুন কুড়ি টাকার নোট", "twenty_old": "পুরানো কুড়ি টাকার নোট",
        "fifty_new": "নতুন পঞ্চাশ টাকার নোট", "fifty_old": "পুরানো পঞ্চাশ টাকার নোট",
        "hundred_new": "নতুন একশ টাকার নোট", "hundred_old": "পুরানো একশ টাকার নোট",
        "two_hundred": "দুইশ টাকার নোট", "five_hundred": "পাঁচশ টাকার নোট",
        "warning_uncertain": "সতর্কতা! মডেলটি নিশ্চিত নয়, আবার স্ক্যান করুন।",
    },
    "mr-IN": {
        "background": "कोणतीही नोट आढळली नाही. कृपया स्पष्टपणे दाखवा.",
        "ten_new": "नवीन दहा रुपयांची नोट", "ten_old": "जुनी दहा रुपयांची नोट",
        "twenty_new": "नवीन वीस रुपयांची नोट", "twenty_old": "जुनी वीस रुपयांची नोट",
        "fifty_new": "नवीन पन्नास रुपयांची नोट", "fifty_old": "जुनी पन्नास रुपयांची नोट",
        "hundred_new": "नवीन शंभर रुपयांची नोट", "hundred_old": "जुनी शंभर रुपयांची नोट",
        "two_hundred": "दोनशे रुपयांची नोट", "five_hundred": "पाचशे रुपयांची नोट",
        "warning_uncertain": "सावधान! मॉडेलला खात्री नाही, कृपया पुन्हा स्कॅन करा.",
    },
    "kn-IN": {
        "background": "ಯಾವುದೇ ನೋಟು ಪತ್ತೆಯಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾಗಿ ತೋರಿಸಿ.",
        "ten_new": "ಹೊಸ ಹತ್ತು ರೂಪಾಯಿ ನೋಟು", "ten_old": "ಹಳೆಯ ಹತ್ತು ರೂಪಾಯಿ ನೋಟು",
        "twenty_new": "ಹೊಸ ಇಪ್ಪತ್ತು ರೂಪಾಯಿ ನೋಟು", "twenty_old": "ಹಳೆಯ ಇಪ್ಪತ್ತು ರೂಪಾಯಿ ನೋಟು",
        "fifty_new": "ಹೊಸ ಐವತ್ತು ರೂಪಾಯಿ ನೋಟు", "fifty_old": "ಹಳೆಯ ಐವತ್ತು ರೂಪಾಯಿ ನೋಟు",
        "hundred_new": "ಹೊಸ ನೂರು ರೂಪಾಯಿ ನೋಟు", "hundred_old": "ಹಳೆಯ ನೂರు ರೂಪಾಯಿ ನೋಟు",
        "two_hundred": "ಇನ್ನೂరు ರೂಪಾಯಿ ನೋಟు", "five_hundred": "ಐನೂరు ರೂಪಾಯಿ ನೋಟు",
        "warning_uncertain": "ಎಚ್ಚరిಕೆ! ಮಾದరి ಖಚಿತವాగిಲ್ಲ, ದయవిట్టు మళ్ళీ ಸ್ಕాన్ మాడి.",
    },
}

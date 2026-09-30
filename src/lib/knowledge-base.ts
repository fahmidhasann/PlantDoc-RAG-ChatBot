export interface PathologyChunk {
  id: string;
  page: number;
  chapter: string;
  topic: string;
  content: string;
  pathogen: string;
  symptoms: string[];
  controls: string[];
}

export const FALLBACK_PATHOLOGY_KNOWLEDGE: PathologyChunk[] = [
  {
    id: "chunk_late_blight",
    page: 421,
    chapter: "Chapter 11: Plant Diseases Caused by Oomycetes",
    topic: "Late Blight of Potato and Tomato",
    pathogen: "Phytophthora infestans (Mont.) de Bary",
    symptoms: [
      "Water-soaked dark lesions on leaves rapidly expanding into purplish-black necrotic areas",
      "White fluffy mildew on the underside of leaves during high humidity",
      "Brownish dry rot on potato tubers with granular dry surface",
      "Dark greasy lesions on tomato stems and green fruit"
    ],
    controls: [
      "Preventive applications of copper fungicides or Mancozeb",
      "Systemic fungicides such as Metalaxyl / Mefenoxam or Dimethomorph",
      "Planting certified disease-free tubers and resistant cultivars (e.g., Defender)",
      "Destroying volunteer plants and cull piles to eliminate inoculum sources",
      "Proper field drainage and avoiding overhead irrigation"
    ],
    content: `Late blight of potato and tomato, caused by the oomycete Phytophthora infestans, is historically the most devastating plant disease, responsible for the Irish potato famine of the 1840s. Symptoms appear as water-soaked irregular spots on leaves that rapidly enlarge, turn brown-black, and blight whole leaves within days. Under moist conditions, a delicate whitish mildew forms at the margin of lesions on the leaf underside consisting of sporangiophores and sporangia. On tubers, irregular, purplish or brownish blotches appear, penetrating into the flesh as a dry, reddish-brown granular rot. Control relies on resistant cultivars, destruction of infected cull piles, certified seed tubers, and prophylactic fungicide sprays (Mancozeb, Chlorothalonil) combined with systemic protectants (Metalaxyl).`
  },
  {
    id: "chunk_bacterial_canker",
    page: 638,
    chapter: "Chapter 12: Plant Diseases Caused by Prokaryotes",
    topic: "Bacterial Canker of Tomato",
    pathogen: "Clavibacter michiganensis subsp. michiganensis",
    symptoms: [
      "Unilateral wilting of leaflets where one side of a leaflet wilts first",
      "White blister-like spots on green fruit turning into bird's-eye spots with dark centers and white halos",
      "Yellowish-white vascular discoloration turning dark brown and necrotic inside stems",
      "Open stem cankers in advanced stages"
    ],
    controls: [
      "Use of certified pathogen-free seeds treated with hot water (50°C for 25 min) or hydrochloric acid",
      "Strict greenhouse sanitation and disinfection of stakes, tools, and flats with quaternary ammonium or 10% bleach",
      "Crop rotation with non-solanaceous crops for at least 3 years",
      "Preventive sprays of copper bactericides mixed with mancozeb (provides synergism)"
    ],
    content: `Bacterial canker of tomato is caused by the Gram-positive coryneform bacterium Clavibacter michiganensis subsp. michiganensis. The pathogen is vascular and seed-borne. Typical early symptoms include marginal necrosis of leaves, often unilateral (affecting one side of a compound leaf or leaflet first). Stems develop light-colored streaks that later crack open into longitudinal cankers. When stems are split open, yellowish vascular discolouration is evident, eventually degenerating into a brown, hollow pith. Fruit display characteristic 'bird's-eye' spots—small white blisters with a dark central crust. Disease management requires pathogen-free seed, seed thermotherapy, sanitation of cutting tools, 3-year crop rotation, and copper-mancozeb tank mixes.`
  },
  {
    id: "chunk_rice_blast",
    page: 495,
    chapter: "Chapter 11: Plant Diseases Caused by Ascomycetes",
    topic: "Rice Blast Disease",
    pathogen: "Magnaporthe oryzae (anamorph Pyricularia oryzae)",
    symptoms: [
      "Spindle-shaped or diamond-shaped lesions on leaves with gray/white centers and brownish borders",
      "Neck blast causing blackening of the panicle neck and breaking of seed heads",
      "Node blast causing rotting and snapping of stem joints",
      "Empty, unfilled grains (white heads)"
    ],
    controls: [
      "Use of resistant varieties with multiple Pi resistance genes",
      "Balanced nitrogen fertilization (avoiding excess nitrogen which predisposes tissue to infection)",
      "Seed treatment with Tricyclazole or Benomyl",
      "Timely foliar fungicide application (Tricyclazole, Azoxystrobin, or Isoprothiolane) at heading stage"
    ],
    content: `Rice blast, caused by the fungus Magnaporthe oryzae, is the most economically destructive disease of cultivated rice worldwide. The fungus attacks all aerial organs. Leaf lesions start as small water-soaked spots, expanding into characteristic spindle-shaped (elliptic) lesions with grayish centers and brown reddish margins. The most destructive phase is neck blast (rotten neck), where the fungus attacks the node immediately below the panicle, turning it blackish-brown and causing the entire panicle to collapse and remain sterile ('white heads'). Control strategies include gene deployment through resistant cultivars, avoiding heavy applications of nitrogen fertilizers, water management, and seed/foliar fungicide applications using melanin biosynthesis inhibitors like tricyclazole.`
  },
  {
    id: "chunk_powdery_mildew",
    page: 462,
    chapter: "Chapter 11: Plant Diseases Caused by Ascomycetes",
    topic: "Powdery Mildew of Cereals, Grapes, and Cucurbits",
    pathogen: "Blumeria graminis / Erysiphe cichoracearum / Podosphaera spp.",
    symptoms: [
      "White to grayish talcum-powder-like patches of superficial mycelium and conidia on upper leaf surfaces",
      "Chlorosis and premature senescence of infected foliage",
      "Stunted growth, fruit distortion, and reduced photosynthetic efficiency",
      "Tiny black spherical cleistothecia/chasmothecia embedded in mycelium late in the season"
    ],
    controls: [
      "Elemental sulfur dust or wettable sulfur (organic standard)",
      "Sterol biosynthesis inhibitors (triazoles: Tebuconazole, Propiconazole)",
      "Potassium bicarbonate or neem oil bio-fungicides",
      "Resistant host varieties (e.g. mlo mutant barley, resistant cucurbits)",
      "Improved canopy airflow through pruning and wider spacing"
    ],
    content: `Powdery mildews are obligate biotrophic ascomycetes attacking hundreds of monocot and dicot hosts. Unlike downy mildews, powdery mildew fungi are predominantly ectoparasitic, developing an extensive superficial white mycelial mat on the epidermis and absorbing nutrients via haustoria inside epidermal cells. Conidia do not require free liquid water to germinate, thriving in warm, dry weather with high relative humidity. Symptoms include white powdery colonies that coalesce over leaves, stems, and fruits, causing curling, yellowing, and necrosis. Control combines cultural management (canopy thinning for air circulation), biological agents (Ampelomyces quisqualis), sulfur sprays, and systemic fungicides (triazoles, strobilurins).`
  },
  {
    id: "chunk_fusarium_wilt",
    page: 542,
    chapter: "Chapter 11: Plant Diseases Caused by Ascomycetes and Deuteromycetes",
    topic: "Fusarium Wilt of Banana (Panama Disease) and Solanaceae",
    pathogen: "Fusarium oxysporum (f. sp. cubense, f. sp. lycopersici)",
    symptoms: [
      "Progressive yellowing and wilting of lower leaves advancing upward",
      "Vascular browning clearly visible when stem is sectioned longitudinally",
      "Stunting and unilateral leaf collapse",
      "Persistent chlamydospores surviving in soil for over 20-30 years"
    ],
    controls: [
      "Strict quarantine to prevent movement of infested soil or suckers",
      "Planting resistant cultivars (e.g., Cavendish against Race 1; somaclonal variants against TR4)",
      "Soil solarization and biocontrol agents (Trichoderma harzianum, Pseudomonas fluorescens)",
      "Crop rotation with non-hosts (suppressive soils)"
    ],
    content: `Fusarium wilt, caused by the soil-borne fungus Fusarium oxysporum, is a vascular wilt disease affecting hundreds of plant species with specialized forms (formae speciales). The fungus enters roots via wounds or root tips, colonizes xylem vessels, and produces microconidia that are carried upward in the transpiration stream. Gels, tyloses, and fungal mycelium clog the vessels, inducing severe water stress, yellowing, wilting, and vascular browning. Chlamydospores survive in soil for decades, making chemical soil treatment ineffective. Management requires resistant varieties, strict phytosanitary measures against tropical race 4 (TR4 in banana), and biological control with antagonistic rhizobacteria.`
  },
  {
    id: "chunk_citrus_greening",
    page: 651,
    chapter: "Chapter 12: Plant Diseases Caused by Prokaryotes",
    topic: "Huanglongbing (HLB) / Citrus Greening",
    pathogen: "Candidatus Liberibacter asiaticus (vectored by Asian Citrus Psyllid, Diaphorina citri)",
    symptoms: [
      "Asymmetric blotchy mottle chlorosis on leaves (not crossing main vein uniformly)",
      "Yellow shoots in healthy green canopy ('huanglongbing')",
      "Lopsided, bitter, poorly colored fruit with inverted coloration (ripening from stem end, remaining green at bottom)",
      "Dieback of twigs and eventual tree death within 3-5 years"
    ],
    controls: [
      "Aggressive vector control targeting Asian citrus psyllid with systemic insecticides (imidacloprid)",
      "Planting disease-free certified nursery stock grown in insect-proof screenhouses",
      "Immediate eradication of infected symptomatic trees to reduce bacterial reservoir",
      "Nutritional foliar sprays and antibiotics (oxytetracycline injection where registered)"
    ],
    content: `Huanglongbing (HLB), or citrus greening, is the most destructive citrus disease globally. Caused by the uncultured, phloem-limited bacterium Candidatus Liberibacter asiaticus and transmitted by the Asian citrus psyllid (Diaphorina citri), it damages all commercially grown citrus cultivars. Symptoms include an asymmetric blotchy mottle pattern on mature leaves, yellow shoot dieback, vein corking, and small, misshapen fruit with abortive seeds and bitter juice. Because no commercial cultivars are immune, management relies on certified pathogen-free trees, comprehensive psyllid suppression, scouting and roguing of infected trees, and strict regional quarantine.`
  }
];

export function searchPathologyKnowledge(query: string, topK: number = 3): PathologyChunk[] {
  const queryLower = query.toLowerCase();
  const queryTerms = queryLower.split(/\W+/).filter(t => t.length > 2);

  const scored = FALLBACK_PATHOLOGY_KNOWLEDGE.map(chunk => {
    let score = 0;
    const contentLower = (chunk.topic + " " + chunk.pathogen + " " + chunk.content + " " + chunk.symptoms.join(" ") + " " + chunk.controls.join(" ")).toLowerCase();

    for (const term of queryTerms) {
      if (chunk.topic.toLowerCase().includes(term)) score += 5;
      if (chunk.pathogen.toLowerCase().includes(term)) score += 6;
      if (chunk.symptoms.some(s => s.toLowerCase().includes(term))) score += 4;
      if (chunk.controls.some(c => c.toLowerCase().includes(term))) score += 4;
      if (contentLower.includes(term)) score += 1;
    }

    return { chunk, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.chunk);
}

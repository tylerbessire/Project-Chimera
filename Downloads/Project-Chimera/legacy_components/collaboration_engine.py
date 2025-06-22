# ==============================================================================
# FILE: collaboration_engine.py (v7.0 - Studio Grade Update)
# ==============================================================================
#
# MAJOR UPDATES:
# - Drastically improved AI prompts based on professional production concepts.
# - Defined a detailed JSON schema for the mashup "recipe".
# - Luna (OpenAI) now acts as the Creative Director, focusing on high-level
#   narrative, structure, and emotional journey.
# - Claude (Anthropic) is now the Audio Engineer, translating Luna's vision
#   into a precise technical recipe with specific parameters for the
#   new RealAudioEngine (EQ, compression, sidechain, effects).
#
# ==============================================================================

import os
import json
import logging
from openai import OpenAI
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class CollaborationEngine:
    """
    Manages the creative collaboration between Luna (Creative Director) and
    Claude (Audio Engineer) to produce a professional mashup recipe.
    """
    def __init__(self):
        self.luna_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.claude_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        if not self.luna_client.api_key or not self.claude_client.api_key:
            raise EnvironmentError("Both OPENAI_API_KEY and ANTHROPIC_API_KEY must be set.")

    def generate_recipe(self, briefs: list, mashup_style: str, progress_callback=None):
        """
        Orchestrates organic real-time AI collaboration between Luna and Claude.
        """
        print("🤝 Starting AI Collaboration...")
        if progress_callback:
            progress_callback("🤝 Starting AI Collaboration...", "ai_collaboration", 10)
        
        # Real-time collaboration - Luna and Claude discuss the mashup organically
        print("🎤 Luna (Creative Director) and 🎛️ Claude (Audio Engineer) collaborating...")
        if progress_callback:
            progress_callback("🎤🎛️ Luna and Claude collaborating in real-time...", "collaboration", 50)
        
        recipe = self._organic_collaboration(briefs, mashup_style)
        
        print("✅ AI collaboration complete. Recipe generated.")
        if progress_callback:
            progress_callback("✅ AI collaboration complete. Recipe generated.", "ai_complete", 90)
        return recipe

    def _organic_collaboration(self, briefs, mashup_style):
        """Luna and Claude collaborate organically on the mashup in real-time."""
        song_details = "\n\n".join([
            f"Song {chr(65+i)}: {b['title']} - {b['tempo']:.1f} BPM, Key: {b['estimated_key']}"
            for i, b in enumerate(briefs)
        ])
        
        # Start Luna's creative response
        luna_prompt = f"""You are Luna, a creative music producer. You're about to collaborate with Claude (audio engineer) on a {mashup_style} mashup of these songs:

{song_details}

Start the collaboration naturally - share your initial creative vision for how these songs could work together. Be specific about what excites you about this combination. Don't use rigid templates, just speak naturally about the creative possibilities you see."""

        luna_response = self.luna_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": luna_prompt}]
        ).choices[0].message.content

        print(f"🎤 Luna: {luna_response}")

        # Claude responds to Luna's vision
        claude_prompt = f"""You are Claude, a precise audio engineer. Luna just shared her creative vision:

"{luna_response}"

Respond to Luna naturally - build on her ideas with your technical perspective. Suggest specific engineering approaches that would bring her vision to life. Keep the conversation flowing naturally."""

        claude_response = self.claude_client.messages.create(
            model="claude-4-sonnet-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": claude_prompt}]
        ).content[0].text

        print(f"🎛️ Claude: {claude_response}")

        # Final technical recipe generation
        recipe_prompt = f"""Based on this collaboration between Luna and Claude:

Luna's Vision: "{luna_response}"
Claude's Technical Response: "{claude_response}"

Create the final technical recipe JSON that brings their collaborative vision to life. Use their actual ideas, not templates."""

        recipe_response = self.claude_client.messages.create(
            model="claude-4-sonnet-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": recipe_prompt}]
        ).content[0].text

        # Extract JSON from Claude's response
        try:
            json_start = recipe_response.find('{')
            json_end = recipe_response.rfind('}') + 1
            json_str = recipe_response[json_start:json_end]
            return json.loads(json_str)
        except:
            # Fallback to structured approach if organic fails
            return self._luna_create_storyboard(briefs, mashup_style)

    def _luna_create_storyboard(self, briefs, mashup_style):
        """Generates the creative storyboard using Luna (GPT-4o-mini)."""
        song_details = "\n\n".join([
            f"--- Song {chr(65+i)} ---\n"
            f"Title: {b['title']}\n"
            f"Tempo: {b['tempo']:.2f} BPM\n"
            f"Key: {b['estimated_key']} ({b['camelot_key']})\n"
            f"Structure: {json.dumps([s['label'] for s in b['structural_segments']])}"
            for i, b in enumerate(briefs)
        ])
        
        prompt = f"""
        You are Luna, a world-class music producer and creative director, like a blend of Avicii and Kill mR Dj. Your task is to design a creative vision for a mashup of two songs.

        Your mashup style is: **{mashup_style}**

        Here are the songs:
        {song_details}

        **Your Task:**
        1.  **Concept:** Invent a creative concept, title, and emotional journey for the mashup. What story will it tell?
        2.  **Roles:** Decide which song provides the primary instrumental foundation (Song A) and which provides the lead vocal (Song B).
        3.  **Structure:** Create a detailed, section-by-section storyboard for the mashup. Use the structural labels from the analysis (e.g., "Intro", "Verse", "Chorus"). Be specific about which elements from each song to use in each section. For example, "Intro: Use the atmospheric pads from Song A's intro, tease a single vocal phrase from Song B's first verse."
        4.  **Energy Flow:** Describe the energy level (1-10) for each section to create a dynamic journey with builds, drops, and breakdowns.

        **Output Format:**
        Produce a JSON object with the following structure. Do not include any text outside the JSON block.
        {{
            "mashup_title": "Your Creative Title",
            "concept": "Your detailed emotional and narrative concept.",
            "roles": {{
                "instrumental_track": "Title of Song A",
                "vocal_track": "Title of Song B"
            }},
            "storyboard": [
                {{
                    "section_label": "Intro",
                    "energy_level": 2,
                    "description": "Start with the filtered synth melody from the instrumental track's intro. Bring in the first line of the vocal track's verse, heavily reverbed and delayed, as a foreshadowing element."
                }},
                {{
                    "section_label": "Verse 1",
                    "energy_level": 5,
                    "description": "The full beat from the instrumental track's verse comes in. Layer the complete first verse from the vocal track on top. Keep the mix clean and focused on the vocal."
                }},
                //... more sections ...
            ]
        }}
        """
        response = self.luna_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _claude_create_technical_recipe(self, storyboard, briefs):
        """Generates the detailed technical recipe using Claude (Haiku)."""
        prompt = f"""
        You are Claude, a meticulous audio engineer inspired by the clean, powerful sound of producers like Dada Life. You have been given a creative storyboard from your producer, Luna. Your job is to translate her vision into a precise technical recipe that the `RealAudioEngine` can execute.

        **LUNA'S STORYBOARD:**
        ```json
        {json.dumps(storyboard, indent=2)}
        ```

        **AVAILABLE AUDIO STEMS & DATA:**
        ```json
        {json.dumps(briefs, indent=2)}
        ```

        **Your Task:**
        For EACH section in Luna's storyboard, create a detailed "layer_cake" object describing exactly how to build it. A "layer_cake" is an array of audio layers.

        **Key Engineering Principles:**
        1.  **EQ for Clarity (Vocal Pocket):** When vocals are present, apply a gentle subtractive EQ cut to the instrumental elements in the 300Hz-3kHz range to create space.
        2.  **Sidechain Compression for Punch:** For high-energy sections, use sidechain compression on bass and pads, triggered by the kick drum, to create a rhythmic pump.
        3.  **Effects for Cohesion & Style:** Use the *same* reverb settings on both vocal and instrumental elements within a section to "glue" them together in the same virtual space. Use filters for transitions and energy builds.
        4.  **Dada Life 'Fatness':** For the highest energy sections, suggest a 'master_effects' chain with saturation and heavy compression.

        **Output Format:**
        Produce a single JSON object. This is the final recipe. Do not add any text outside the JSON block.

        **JSON Schema:**
        {{
          "mashup_title": "From Luna's Storyboard",
          "target_bpm": 126.0, // Choose a good target BPM, maybe the average or the instrumental's.
          "roles": {{ ... from storyboard ... }},
          "storyboard": {{ ... from storyboard ... }}, // Copy Luna's storyboard here for reference
          "source_files": {{
            "instrumental_track": {{ "path": "path/to/instrumental/analysis.json" }},
            "vocal_track": {{ "path": "path/to/vocal/analysis.json" }}
          }},
          "sections": [
            {{
              "section_label": "Intro", // From storyboard
              "layer_cake": [
                {{
                  "source_track": "instrumental_track",
                  "stem": "other", // (vocals, drums, bass, other)
                  "source_segment_label": "Intro", // Segment from analysis to use
                  "effects_chain": [
                    {{ "effect": "lowpass_filter", "cutoff_hz": 800 }},
                    {{ "effect": "reverb", "wet_level": 0.4, "room_size": 0.8 }}
                  ]
                }},
                {{
                  "source_track": "vocal_track",
                  "stem": "vocals",
                  "source_segment_label": "Verse", // Can be different
                  "time_slice_ms": [0, 4000], // Use first 4 seconds of the segment
                  "effects_chain": [
                    {{ "effect": "reverb", "wet_level": 0.6, "room_size": 0.8 }},
                    {{ "effect": "delay", "feedback": 0.5, "mix": 0.4 }}
                  ]
                }}
              ]
            }}
            // ... more sections
          ],
          "master_effects_chain": [
             {{ "effect": "compressor", "threshold_db": -12, "ratio": 2.5, "attack_ms": 5, "release_ms": 150 }},
             {{ "effect": "limiter", "threshold_db": -1.0 }}
          ]
        }}
        """
        message = self.claude_client.messages.create(
            model="claude-4-sonnet-20250514",
            max_tokens=8192,  # Increased limit for complex recipes
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text

        # Clean Claude's response to extract only the JSON
        try:
            json_start = message.find('{')
            json_end = message.rfind('}') + 1
            json_str = message[json_start:json_end]
            
            # Try to parse the JSON
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            logger.warning(f"JSON parsing failed, attempting repair. Error: {e}")
            
            # Find the last complete object before the error
            json_start = message.find('{')
            if json_start == -1:
                raise Exception("No JSON object found in Claude's response")
            
            # Count braces to find a complete JSON object
            brace_count = 0
            last_complete = json_start
            
            for i, char in enumerate(message[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete = i + 1
                        break
            
            try:
                json_str = message[json_start:last_complete]
                result = json.loads(json_str)
                logger.info("Successfully repaired JSON by finding complete object")
                return result
            except Exception as repair_error:
                # Save the problematic response for debugging
                logger.error(f"Claude JSON parsing failed completely. Raw response length: {len(message)}")
                logger.error(f"Original error: {e}")
                logger.error(f"Repair error: {repair_error}")
                logger.error(f"JSON extraction attempt: {json_str[:1000]}...")
                raise Exception(f"Claude (Anthropic) failed to produce a valid JSON recipe: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing Claude response: {e}")
            raise Exception(f"Claude (Anthropic) failed to produce a valid JSON recipe: {e}")

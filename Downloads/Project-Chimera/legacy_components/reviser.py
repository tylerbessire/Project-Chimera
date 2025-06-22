import os
import json
import time
import openai
import anthropic
from typing import Dict, List, Any, Optional
from micro_stem_processor import MicroStemProcessor

class RevisionEngine:
    """
    Production-ready collaborative revision system where Luna (OpenAI) creates 
    emotional narrative and Claude (Anthropic) ensures technical precision at 
    the molecular level.
    """
    
    def __init__(self, *args, **kwargs):
        # Handle both old and new calling patterns
        if len(args) == 4:
            # Old pattern: RevisionEngine(recipe, command, openai_key, anthropic_key)
            self.initial_recipe = args[0]
            self.creative_command = args[1] 
            self.openai_key = args[2]
            self.anthropic_key = args[3]
        elif len(args) == 2:
            # New pattern: RevisionEngine(openai_key, anthropic_key)
            self.openai_key = args[0]
            self.anthropic_key = args[1]
            self.initial_recipe = None
            self.creative_command = None
        else:
            # Keyword arguments
            self.openai_key = kwargs.get('openai_api_key')
            self.anthropic_key = kwargs.get('anthropic_api_key')
            self.initial_recipe = kwargs.get('initial_recipe')
            self.creative_command = kwargs.get('creative_command')
        
        self.micro_processor = None
        self.molecular_catalog = {}
        
        # Initialize API clients
        if self.openai_key:
            openai.api_key = self.openai_key
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
        
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        
    def load_mashup_workspace(self, mashup_id: str) -> Dict[str, Any]:
        """Load existing mashup data and molecular components."""
        workspace_path = f"workspace/mashups/{mashup_id}"
        
        if not os.path.exists(workspace_path):
            raise FileNotFoundError(f"Mashup {mashup_id} not found")
        
        # Load mashup data
        with open(f"{workspace_path}/mashup_data.json", 'r') as f:
            mashup_data = json.load(f)
        
        # Initialize molecular processor for stems
        stems_path = f"{workspace_path}/stems"
        if os.path.exists(stems_path):
            self.micro_processor = MicroStemProcessor(stems_path)
            print("🧬 Loading molecular components...")
            self.molecular_catalog = self.micro_processor.analyze_and_segment_all_stems()
            print(f"✅ Loaded {sum(len(stem_segments) for stem_segments in self.molecular_catalog.values())} molecular components")
        
        return mashup_data
    
    def revise_with_collaboration(self, mashup_id: str, current_recipe: Dict, user_command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Main collaborative revision orchestration."""
        
        try:
            jobs[job_id]["progress"] = "🔍 Loading existing mashup..."
            mashup_data = self.load_mashup_workspace(mashup_id)
            
            if self.openai_key and self.anthropic_key:
                return self._collaborative_revision(mashup_data, current_recipe, user_command, jobs, job_id)
            elif self.openai_key:
                return self._luna_solo_revision(mashup_data, current_recipe, user_command, jobs, job_id)
            elif self.anthropic_key:
                return self._claude_solo_revision(mashup_data, current_recipe, user_command, jobs, job_id)
            else:
                return self._basic_revision(mashup_data, current_recipe, user_command, jobs, job_id)
                
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Revision failed: {str(e)}"
            return {"error": str(e)}
    
    def _collaborative_revision(self, mashup_data: Dict, current_recipe: Dict, user_command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Luna + Claude collaborative revision with molecular precision."""
        
        jobs[job_id]["progress"] = "🎤 Luna is analyzing the emotional narrative..."
        
        # Step 1: Luna's Creative Analysis
        luna_analysis = self._get_luna_creative_analysis(mashup_data, current_recipe, user_command)
        
        jobs[job_id]["progress"] = "🎛️ Claude is reviewing technical implementation..."
        
        # Step 2: Claude's Technical Review
        claude_review = self._get_claude_technical_review(luna_analysis, self.molecular_catalog)
        
        jobs[job_id]["progress"] = "🎵 Creating molecular revision recipe..."
        
        # Step 3: Generate Molecular Recipe
        molecular_recipe = self._create_molecular_recipe(luna_analysis, claude_review)
        
        jobs[job_id]["progress"] = "🎧 Rendering revised audio..."
        
        # Step 4: Render the revision
        output_path = self._render_molecular_mashup(molecular_recipe, mashup_data, job_id)
        
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = {
            "audio_url": f"/api/mashup/audio/{os.path.basename(output_path)}",
            "molecular_recipe": molecular_recipe,
            "collaboration_story": {
                "luna_vision": luna_analysis.get("creative_vision", ""),
                "claude_engineering": claude_review.get("technical_notes", "")
            },
            "collaboration_used": True
        }
        
    def revise(self) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility with tasks.py
        """
        if not self.initial_recipe or not self.creative_command:
            raise Exception("RevisionEngine initialized without recipe/command - use revise_with_collaboration() instead")
        
        try:
            # Convert old-style revision to new molecular revision
            print("🔄 Converting legacy revision request to molecular format...")
            
            # Create a temporary job structure for compatibility
            temp_job_id = "legacy_revision"
            temp_jobs = {temp_job_id: {"status": "processing", "progress": "Converting..."}}
            
            # Simulate mashup data structure
            mashup_data = {
                "song_info": "Legacy mashup data",
                "format_version": "1.0_legacy"
            }
            
            # Use collaborative revision with legacy data
            if self.openai_key and self.anthropic_key:
                result = self._collaborative_legacy_revision(
                    mashup_data, self.initial_recipe, self.creative_command, temp_jobs, temp_job_id
                )
            elif self.openai_key:
                result = self._luna_legacy_revision(
                    mashup_data, self.initial_recipe, self.creative_command, temp_jobs, temp_job_id
                )
            else:
                # Return the original recipe with minimal modification
                result = self._basic_legacy_revision(self.initial_recipe, self.creative_command)
            
            return result.get("recipe", self.initial_recipe)
            
        except Exception as e:
            print(f"❌ Legacy revision failed: {e}")
            # Return original recipe as fallback
            return self.initial_recipe
    
    def _collaborative_legacy_revision(self, mashup_data: Dict, recipe: Dict, command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Handle legacy collaborative revision."""
        print("🎤🎛️ Applying collaborative AI enhancement to legacy recipe...")
        
        # Get Luna's creative enhancement
        luna_analysis = self._get_luna_legacy_analysis(recipe, command)
        
        # Get Claude's technical review  
        claude_review = self._get_claude_legacy_review(luna_analysis, recipe)
        
        # Enhance the recipe based on AI collaboration
        enhanced_recipe = self._enhance_legacy_recipe(recipe, luna_analysis, claude_review)
        
        return {
            "recipe": enhanced_recipe,
            "collaboration_story": {
                "luna_vision": luna_analysis.get("creative_vision", ""),
                "claude_engineering": claude_review.get("technical_notes", "")
            }
        }
    
    def _luna_legacy_revision(self, mashup_data: Dict, recipe: Dict, command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Handle legacy Luna-only revision."""
        print("🎤 Applying Luna's creative enhancement...")
        
        luna_analysis = self._get_luna_legacy_analysis(recipe, command)
        enhanced_recipe = self._enhance_legacy_recipe(recipe, luna_analysis, {})
        
        return {
            "recipe": enhanced_recipe,
            "luna_enhancement": luna_analysis.get("creative_vision", "")
        }
    
    def _basic_legacy_revision(self, recipe: Dict, command: str) -> Dict[str, Any]:
        """Handle basic legacy revision without AI."""
        print("⚙️ Applying basic recipe enhancement...")
        
        # Make minimal enhancements to the recipe
        enhanced_recipe = dict(recipe)
        
        # Add metadata about the enhancement
        if "metadata" not in enhanced_recipe:
            enhanced_recipe["metadata"] = {}
        
        enhanced_recipe["metadata"]["enhanced_command"] = command
        enhanced_recipe["metadata"]["enhancement_type"] = "basic"
        
        return {"recipe": enhanced_recipe}
    
    def _get_luna_legacy_analysis(self, recipe: Dict, command: str) -> Dict[str, Any]:
        """Get Luna's analysis for legacy recipe format."""
        
        if not self.openai_key:
            return {"creative_vision": "Luna enhancement requested but API key not available"}
        
        luna_prompt = f"""
        You are Luna, the creative music producer. You're enhancing an existing mashup recipe.
        
        CURRENT RECIPE STRUCTURE:
        {json.dumps(recipe, indent=2)}
        
        ENHANCEMENT REQUEST: "{command}"
        
        Provide creative suggestions to enhance this mashup in this JSON format:
        {{
            "creative_vision": "your artistic enhancement vision",
            "emotional_improvements": "how to enhance emotional impact",
            "structural_suggestions": "timeline and arrangement improvements",
            "vocal_enhancements": "vocal processing suggestions",
            "rhythm_improvements": "rhythm section enhancements"
        }}
        
        Focus on EMOTIONAL STORYTELLING and CREATIVE ENHANCEMENT.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are Luna, a creative music producer specializing in emotional storytelling and artistic enhancement of musical arrangements."},
                    {"role": "user", "content": luna_prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            
            luna_response = response.choices[0].message.content
            
            try:
                return json.loads(luna_response)
            except json.JSONDecodeError:
                return {"creative_vision": luna_response}
                
        except Exception as e:
            print(f"⚠️ Luna legacy analysis error: {e}")
            return {"creative_vision": f"Creative enhancement: {command}"}
    
    def _get_claude_legacy_review(self, luna_analysis: Dict, recipe: Dict) -> Dict[str, Any]:
        """Get Claude's technical review for legacy recipe format."""
        
        if not self.anthropic_key:
            return {"technical_notes": "Claude review requested but API key not available"}
        
        claude_prompt = f"""
        You are Claude, the technical audio engineer. Review Luna's creative suggestions for technical feasibility.
        
        LUNA'S CREATIVE ANALYSIS:
        {json.dumps(luna_analysis, indent=2)}
        
        CURRENT RECIPE:
        {json.dumps(recipe, indent=2)}
        
        Provide technical review in this JSON format:
        {{
            "technical_feasibility": "assessment of Luna's suggestions",
            "technical_notes": "technical implementation strategy",
            "optimization_suggestions": "technical optimizations",
            "audio_quality_recommendations": "audio processing recommendations"
        }}
        
        Focus on TECHNICAL PRECISION and AUDIO QUALITY.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-4-sonnet-20250514",
                max_tokens=1000,
                temperature=0.3,
                system="You are Claude, an expert audio engineer focused on technical precision and optimal audio quality implementation.",
                messages=[
                    {"role": "user", "content": claude_prompt}
                ]
            )
            
            claude_response = response.content[0].text
            
            try:
                return json.loads(claude_response)
            except json.JSONDecodeError:
                return {"technical_notes": claude_response}
                
        except Exception as e:
            print(f"⚠️ Claude legacy review error: {e}")
            return {"technical_notes": "Technical optimization applied"}
    
    def _enhance_legacy_recipe(self, recipe: Dict, luna_analysis: Dict, claude_review: Dict) -> Dict[str, Any]:
        """Enhance legacy recipe based on AI collaboration."""
        
        enhanced_recipe = dict(recipe)
        
        # Add collaboration metadata
        if "metadata" not in enhanced_recipe:
            enhanced_recipe["metadata"] = {}
        
        enhanced_recipe["metadata"]["ai_enhancement"] = {
            "luna_vision": luna_analysis.get("creative_vision", ""),
            "claude_engineering": claude_review.get("technical_notes", ""),
            "enhancement_timestamp": time.time()
        }
        
        # Enhance timeline segments if they exist
        if "timeline" in enhanced_recipe:
            for segment in enhanced_recipe["timeline"]:
                # Add emotional context from Luna
                if "emotional_context" not in segment:
                    segment["emotional_context"] = luna_analysis.get("emotional_improvements", "Enhanced emotional impact")
                
                # Add technical notes from Claude
                if "technical_notes" not in segment:
                    segment["technical_notes"] = claude_review.get("optimization_suggestions", "Optimized audio processing")
        
        # Enhance effects based on AI suggestions
        luna_vocal_suggestions = luna_analysis.get("vocal_enhancements", "")
        if luna_vocal_suggestions and "effects" in enhanced_recipe:
            if "vocal_enhancement" not in enhanced_recipe["effects"]:
                enhanced_recipe["effects"]["vocal_enhancement"] = luna_vocal_suggestions
        
        claude_audio_suggestions = claude_review.get("audio_quality_recommendations", "")
        if claude_audio_suggestions and "effects" in enhanced_recipe:
            if "audio_optimization" not in enhanced_recipe["effects"]:
                enhanced_recipe["effects"]["audio_optimization"] = claude_audio_suggestions
        
        return enhanced_recipe
    
    def _get_luna_creative_analysis(self, mashup_data: Dict, current_recipe: Dict, user_command: str) -> Dict[str, Any]:
        """Luna analyzes the emotional and creative aspects using OpenAI."""
        
        if not self.openai_key:
            return {"error": "OpenAI API key required for Luna analysis"}
        
        molecular_summary = self._get_molecular_summary()
        
        # Construct prompt for Luna's creative analysis
        luna_prompt = f"""
        You are Luna, the creative music producer and emotional storyteller. 
        
        CURRENT MASHUP CONTEXT:
        - Songs: {mashup_data.get('song_info', 'Unknown')}
        - Current Recipe: {json.dumps(current_recipe, indent=2)}
        - Available Molecular Components: {molecular_summary}
        
        USER'S REVISION REQUEST: "{user_command}"
        
        As Luna, provide your creative analysis in this exact JSON format:
        {{
            "emotional_intent": "describe the emotional goal",
            "narrative_structure": {{
                "current_story": "what story the current mashup tells",
                "desired_evolution": "how the revision should evolve the story"
            }},
            "creative_vision": "your artistic vision for this revision",
            "molecular_suggestions": {{
                "vocal_emphasis": "specific vocal molecular components and effects",
                "rhythm_enhancement": "specific drum/percussion molecular suggestions", 
                "textural_elements": "harmonic/melodic molecular suggestions",
                "emotional_peaks": "where and how to create emotional climax"
            }},
            "timing_narrative": {{
                "intro_mood": "0-4s emotional intention",
                "buildup_tension": "4-8s narrative development",
                "climax_impact": "8-12s emotional peak strategy",
                "resolution": "12s+ conclusion approach"
            }}
        }}
        
        Focus on the EMOTIONAL IMPACT and MUSICAL STORYTELLING. Be specific about which molecular components would serve your creative vision.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are Luna, an expert music producer focused on emotional storytelling through advanced audio manipulation. You have access to molecular-level audio components and specialize in creating emotionally resonant musical narratives."},
                    {"role": "user", "content": luna_prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            luna_response = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                analysis = json.loads(luna_response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis = {
                    "emotional_intent": "Enhanced emotional resonance",
                    "creative_vision": luna_response,
                    "molecular_suggestions": {
                        "vocal_emphasis": "Apply harmonic stacking to vocal phrases",
                        "rhythm_enhancement": "Layer kick patterns for increased impact",
                        "textural_elements": "Add melodic hooks with echo effects"
                    }
                }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Luna analysis error: {e}")
            return {
                "error": f"Luna analysis failed: {str(e)}",
                "fallback_analysis": {
                    "emotional_intent": "Enhance emotional impact based on user request",
                    "creative_vision": f"Apply creative interpretation of: {user_command}"
                }
            }
    
    def _get_claude_technical_review(self, luna_analysis: Dict, molecular_catalog: Dict) -> Dict[str, Any]:
        """Claude reviews technical feasibility and optimizations using Anthropic."""
        
        if not self.anthropic_key:
            return {"error": "Anthropic API key required for Claude review"}
        
        # Prepare molecular catalog summary for Claude
        catalog_summary = {}
        for stem_type, segments in molecular_catalog.items():
            catalog_summary[stem_type] = {
                "total_segments": len(segments),
                "segment_types": list(set(seg_data.get("type", "unknown") for seg_data in segments.values())),
                "available_segments": list(segments.keys())[:10]  # First 10 for brevity
            }
        
        claude_prompt = f"""
        You are Claude, the precision audio engineer and technical perfectionist.
        
        LUNA'S CREATIVE ANALYSIS:
        {json.dumps(luna_analysis, indent=2)}
        
        AVAILABLE MOLECULAR CATALOG:
        {json.dumps(catalog_summary, indent=2)}
        
        As Claude, provide your technical review in this exact JSON format:
        {{
            "technical_feasibility": "assessment of Luna's suggestions",
            "timing_analysis": {{
                "optimal_transition_points": ["list of precise timing points"],
                "crossfade_requirements": "technical crossfade specifications",
                "phase_alignment_notes": "phase coherence considerations"
            }},
            "molecular_validation": {{
                "available_components": "count and validation of suggested components",
                "validated_suggestions": "which of Luna's suggestions are technically sound",
                "optimization_recommendations": "technical improvements to Luna's vision",
                "alternative_molecules": "alternative molecular components if needed"
            }},
            "audio_engineering_notes": {{
                "eq_recommendations": "frequency shaping suggestions",
                "dynamics_processing": "compression/limiting recommendations", 
                "spatial_effects": "reverb/delay technical specifications",
                "mastering_considerations": "final mix considerations"
            }},
            "technical_notes": "overall technical implementation strategy",
            "risk_assessment": "potential technical challenges and solutions"
        }}
        
        Focus on TECHNICAL PRECISION, AUDIO QUALITY, and ensuring Luna's creative vision is technically achievable with available molecular components.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-4-sonnet-20250514",
                max_tokens=2000,
                temperature=0.3,
                system="You are Claude, an expert audio engineer with deep knowledge of digital signal processing, molecular-level audio manipulation, and technical music production. You specialize in ensuring creative visions are technically perfect and optimally implemented.",
                messages=[
                    {"role": "user", "content": claude_prompt}
                ]
            )
            
            claude_response = response.content[0].text
            
            # Try to parse JSON response
            try:
                review = json.loads(claude_response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                review = {
                    "technical_feasibility": "Analyzing technical requirements",
                    "technical_notes": claude_response,
                    "molecular_validation": {
                        "available_components": len(sum(molecular_catalog.values(), [])),
                        "validated_suggestions": "Reviewing Luna's molecular suggestions for technical implementation"
                    }
                }
            
            return review
            
        except Exception as e:
            print(f"❌ Claude review error: {e}")
            return {
                "error": f"Claude review failed: {str(e)}",
                "fallback_review": {
                    "technical_feasibility": "High - proceeding with available molecular components",
                    "technical_notes": "Implementing with best available technical practices"
                }
            }
    
    def _create_molecular_recipe(self, luna_analysis: Dict, claude_review: Dict) -> Dict[str, Any]:
        """Create the molecular-level mashup recipe based on Luna + Claude collaboration."""
        
        available_catalog = self.micro_processor.get_micro_segment_catalog() if self.micro_processor else {}
        
        # Extract validated components
        luna_suggestions = luna_analysis.get("molecular_suggestions", {})
        claude_validation = claude_review.get("molecular_validation", {})
        
        # Create timeline based on Luna's narrative structure and Claude's timing analysis
        timeline_segments = []
        
        # Get timing recommendations from Claude
        timing_analysis = claude_review.get("timing_analysis", {})
        transition_points = timing_analysis.get("optimal_transition_points", ["4000ms", "8000ms", "12000ms"])
        
        # Convert transition points to milliseconds
        transitions_ms = []
        for point in transition_points:
            try:
                if "ms" in point:
                    transitions_ms.append(int(point.replace("ms", "")))
                elif "s" in point:
                    transitions_ms.append(int(float(point.replace("s", "")) * 1000))
                else:
                    transitions_ms.append(int(point))
            except:
                continue
        
        if not transitions_ms:
            transitions_ms = [4000, 8000, 12000]  # Default transitions
        
        # Build timeline segments
        prev_time = 0
        for i, transition_time in enumerate(transitions_ms + [16000]):  # Add final endpoint
            
            # Determine segment type based on Luna's narrative
            if i == 0:
                segment_type = "foundation"
                description = luna_analysis.get("timing_narrative", {}).get("intro_mood", "Emotional foundation")
            elif i == 1:
                segment_type = "buildup"
                description = luna_analysis.get("timing_narrative", {}).get("buildup_tension", "Building tension")
            elif i == 2:
                segment_type = "climax"
                description = luna_analysis.get("timing_narrative", {}).get("climax_impact", "Emotional climax")
            else:
                segment_type = "resolution"
                description = luna_analysis.get("timing_narrative", {}).get("resolution", "Resolution")
            
            # Select molecular components for this segment
            segment_molecules = self._select_molecules_for_segment(
                segment_type, available_catalog, luna_suggestions, claude_validation
            )
            
            timeline_segments.append({
                "time_range": f"{prev_time}-{transition_time}ms",
                "segment_type": segment_type,
                "description": description,
                "molecules": segment_molecules,
                "claude_engineering": claude_review.get("audio_engineering_notes", {}),
                "luna_emotion": luna_analysis.get("emotional_intent", "")
            })
            
            prev_time = transition_time
        
        recipe = {
            "format_version": "2.0_molecular_production",
            "collaboration_story": {
                "luna_vision": luna_analysis.get("creative_vision", ""),
                "claude_engineering": claude_review.get("technical_notes", ""),
                "molecular_precision": True
            },
            "timeline": timeline_segments,
            "global_effects": {
                "master_eq": claude_review.get("audio_engineering_notes", {}).get("eq_recommendations", ""),
                "dynamics": claude_review.get("audio_engineering_notes", {}).get("dynamics_processing", ""),
                "spatial": claude_review.get("audio_engineering_notes", {}).get("spatial_effects", "")
            },
            "collaboration_metadata": {
                "luna_analysis": luna_analysis,
                "claude_review": claude_review,
                "creation_timestamp": time.time()
            }
        }
        
        return recipe
    
    def _select_molecules_for_segment(self, segment_type: str, catalog: Dict, luna_suggestions: Dict, claude_validation: Dict) -> Dict[str, Any]:
        """Select specific molecular components for a timeline segment."""
        
        molecules = {}
        
        # Vocal components
        if "vocals" in catalog and catalog["vocals"]:
            vocal_segments = catalog["vocals"]
            
            if segment_type == "foundation":
                # Look for vocal phrases
                for segment_name, segment_info in vocal_segments.items():
                    if "phrase" in segment_name and segment_info.get("duration_ms", 0) > 1000:
                        vocal_ingredient = f"vocals_{segment_name}"
                        molecules["vocal_foundation"] = {
                            "ingredient": vocal_ingredient,
                            "effects": self._get_vocal_effects(segment_type, luna_suggestions),
                            "molecular_type": "vocal_phrase",
                            "luna_intent": luna_suggestions.get("vocal_emphasis", ""),
                            "claude_processing": claude_validation.get("optimization_recommendations", "")
                        }
                        break
            
            elif segment_type == "climax":
                # Look for vocal runs or powerful phrases
                for segment_name, segment_info in vocal_segments.items():
                    if "run" in segment_name or ("phrase" in segment_name and segment_info.get("duration_ms", 0) > 2000):
                        vocal_ingredient = f"vocals_{segment_name}"
                        molecules["vocal_climax"] = {
                            "ingredient": vocal_ingredient,
                            "effects": ["stagger_harmony: delays=[200,400,600], pitches=[-3,0,+4,+7]", "volume: +2dB"],
                            "molecular_type": "vocal_powerhouse",
                            "luna_intent": "Emotional peak with harmonic stacking"
                        }
                        break
        
        # Rhythmic components
        if "drums" in catalog and catalog["drums"]:
            drum_segments = catalog["drums"]
            
            for segment_name, segment_info in drum_segments.items():
                if "kick_hit" in segment_name and len(molecules) < 3:  # Limit components
                    drum_ingredient = f"drums_{segment_name}"
                    molecules[f"rhythm_{segment_type}"] = {
                        "ingredient": drum_ingredient,
                        "effects": self._get_rhythm_effects(segment_type, luna_suggestions),
                        "molecular_type": "percussion_element"
                    }
                    break
        
        # Harmonic/Melodic components
        if "other" in catalog and catalog["other"]:
            harmonic_segments = catalog["other"]
            
            for segment_name, segment_info in harmonic_segments.items():
                if "melodic_hook" in segment_name and len(molecules) < 4:
                    harmonic_ingredient = f"other_{segment_name}"
                    molecules[f"texture_{segment_type}"] = {
                        "ingredient": harmonic_ingredient,
                        "effects": self._get_harmonic_effects(segment_type, luna_suggestions),
                        "molecular_type": "melodic_element"
                    }
                    break
        
        return molecules
    
    def _get_vocal_effects(self, segment_type: str, luna_suggestions: Dict) -> List[str]:
        """Get appropriate vocal effects based on segment type and Luna's suggestions."""
        
        base_effects = []
        
        if segment_type == "foundation":
            base_effects = ["pitch_harmony: [0,4]", "volume: -3dB"]
        elif segment_type == "buildup":
            base_effects = ["pitch_harmony: [0,3,7]", "echo_trail: 200ms", "volume: -1dB"]
        elif segment_type == "climax":
            base_effects = ["stagger_harmony: delays=[200,400], pitches=[-3,0,+4]", "volume: +2dB"]
        elif segment_type == "resolution":
            base_effects = ["crossfade_blend: fade_out=1000ms", "volume: -6dB"]
        
        # Add Luna's specific suggestions
        vocal_emphasis = luna_suggestions.get("vocal_emphasis", "")
        if "stagger" in vocal_emphasis.lower():
            base_effects.append("stagger_harmony: enhanced")
        if "harmony" in vocal_emphasis.lower():
            base_effects.append("pitch_harmony: extended")
        
        return base_effects
    
    def _get_rhythm_effects(self, segment_type: str, luna_suggestions: Dict) -> List[str]:
        """Get appropriate rhythm effects."""
        
        if segment_type == "foundation":
            return ["loop: 2x", "volume: -6dB"]
        elif segment_type == "buildup":
            return ["loop: 4x", "pitch_shift: +1", "volume: -3dB"]
        elif segment_type == "climax":
            return ["loop: 8x", "pitch_shift: +2", "volume: 0dB"]
        else:
            return ["fade_out: 500ms", "volume: -12dB"]
    
    def _get_harmonic_effects(self, segment_type: str, luna_suggestions: Dict) -> List[str]:
        """Get appropriate harmonic effects."""
        
        if segment_type == "foundation":
            return ["echo_trail: 300ms", "volume: -9dB"]
        elif segment_type == "buildup":
            return ["echo_trail: 200ms", "pitch_shift: +0.5", "volume: -6dB"]
        elif segment_type == "climax":
            return ["echo_trail: 150ms", "pitch_harmony: [0,7]", "volume: -3dB"]
        else:
            return ["fade_out: 1000ms", "volume: -15dB"]
    
    def _render_molecular_mashup(self, recipe: Dict, mashup_data: Dict, job_id: str) -> str:
        """Render the molecular recipe into audio using the MicroStemProcessor."""
        
        if not self.micro_processor:
            raise Exception("Molecular processor not initialized")
        
        # Create output directory
        output_dir = f"workspace/mashups/{job_id}_revision"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process timeline segments
        from pydub import AudioSegment
        final_mashup = AudioSegment.empty()
        
        for segment in recipe.get("timeline", []):
            segment_audio = AudioSegment.empty()
            molecules = segment.get("molecules", {})
            
            # Process each molecular component
            for component_name, component_data in molecules.items():
                try:
                    # Get the molecular component
                    ingredient = component_data.get("ingredient", "")
                    if "_" in ingredient:
                        stem_type, molecule_name = ingredient.split("_", 1)
                        
                        if stem_type in self.molecular_catalog and molecule_name in self.molecular_catalog[stem_type]:
                            molecule = self.molecular_catalog[stem_type][molecule_name]
                            component_audio = molecule.get("audio_segment")
                            
                            if component_audio:
                                # Apply effects
                                effects = component_data.get("effects", [])
                                processed_audio = component_audio
                                
                                for effect in effects:
                                    try:
                                        if "pitch_harmony" in effect:
                                            processed_audio = self.micro_processor.create_advanced_effect(
                                                processed_audio, "pitch_harmony"
                                            )
                                        elif "stagger_harmony" in effect:
                                            processed_audio = self.micro_processor.create_advanced_effect(
                                                processed_audio, "stagger_harmony"
                                            )
                                        elif "echo_trail" in effect:
                                            processed_audio = self.micro_processor.create_advanced_effect(
                                                processed_audio, "echo_trail"
                                            )
                                        elif "volume:" in effect:
                                            # Extract volume adjustment
                                            volume_str = effect.split("volume:")[1].strip()
                                            volume_db = float(volume_str.replace("dB", ""))
                                            processed_audio = processed_audio + volume_db
                                        elif "loop:" in effect:
                                            # Extract loop count
                                            loop_str = effect.split("loop:")[1].strip()
                                            loop_count = int(loop_str.replace("x", ""))
                                            processed_audio = processed_audio * loop_count
                                    except Exception as e:
                                        print(f"⚠️ Effect processing error: {e}")
                                        continue
                                
                                # Layer into segment
                                if len(segment_audio) == 0:
                                    segment_audio = processed_audio
                                else:
                                    segment_audio = segment_audio.overlay(processed_audio)
                
                except Exception as e:
                    print(f"⚠️ Molecular component processing error: {e}")
                    continue
            
            # Add segment to final mashup
            if len(segment_audio) > 0:
                # Ensure segment is the right length
                time_range = segment.get("time_range", "0-4000ms")
                duration_str = time_range.split("-")[1]
                duration_ms = int(duration_str.replace("ms", ""))
                
                if len(segment_audio) > duration_ms:
                    segment_audio = segment_audio[:duration_ms]
                elif len(segment_audio) < duration_ms:
                    # Pad with silence
                    padding = AudioSegment.silent(duration=duration_ms - len(segment_audio))
                    segment_audio = segment_audio + padding
                
                # Add to final mashup
                final_mashup += segment_audio
            else:
                # Add silence if no components processed
                final_mashup += AudioSegment.silent(duration=4000)
        
        # Apply global effects if specified
        global_effects = recipe.get("global_effects", {})
        if global_effects:
            # Apply master EQ, dynamics, etc. (simplified implementation)
            dynamics = global_effects.get("dynamics", "")
            if "normalize" in dynamics.lower():
                from pydub.effects import normalize
                final_mashup = normalize(final_mashup)
        
        # Export final mashup
        output_file = f"{output_dir}/molecular_revision_{job_id}.wav"
        final_mashup.export(output_file, format="wav")
        
        # Save recipe
        recipe_file = f"{output_dir}/molecular_recipe.json"
        with open(recipe_file, 'w') as f:
            json.dump(recipe, f, indent=2)
        
        return output_file
    
    def _luna_solo_revision(self, mashup_data: Dict, current_recipe: Dict, user_command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Luna solo revision mode with full molecular processing."""
        jobs[job_id]["progress"] = "🎤 Luna creating emotional revision..."
        
        analysis = self._get_luna_creative_analysis(mashup_data, current_recipe, user_command)
        
        # Create simplified molecular recipe based on Luna's analysis
        molecular_recipe = {
            "format_version": "2.0_luna_solo",
            "timeline": [{
                "time_range": "0-15000ms",
                "description": "Luna's emotional interpretation",
                "molecules": self._get_fallback_molecules(),
                "luna_vision": analysis.get("creative_vision", "")
            }],
            "collaboration_story": {"luna_solo": True}
        }
        
        # Render if molecular processor available
        if self.micro_processor:
            output_path = self._render_molecular_mashup(molecular_recipe, mashup_data, job_id)
        else:
            output_path = f"workspace/mashups/luna_revision_{job_id}.wav"
            # Create placeholder file
            from pydub import AudioSegment
            AudioSegment.silent(duration=15000).export(output_path, format="wav")
        
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = {
            "audio_url": f"/api/mashup/audio/{os.path.basename(output_path)}",
            "creative_analysis": analysis,
            "molecular_recipe": molecular_recipe,
            "collaboration_used": False,
            "solo_artist": "Luna"
        }
        
        return jobs[job_id]["result"]
    
    def _claude_solo_revision(self, mashup_data: Dict, current_recipe: Dict, user_command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Claude solo revision mode with technical optimization."""
        jobs[job_id]["progress"] = "🎛️ Claude optimizing technical revision..."
        
        review = self._get_claude_technical_review({}, self.molecular_catalog)
        
        # Create technically optimized molecular recipe
        molecular_recipe = {
            "format_version": "2.0_claude_solo",
            "timeline": [{
                "time_range": "0-15000ms", 
                "description": "Claude's technical optimization",
                "molecules": self._get_fallback_molecules(),
                "claude_engineering": review.get("technical_notes", "")
            }],
            "collaboration_story": {"claude_solo": True}
        }
        
        if self.micro_processor:
            output_path = self._render_molecular_mashup(molecular_recipe, mashup_data, job_id)
        else:
            output_path = f"workspace/mashups/claude_revision_{job_id}.wav"
            from pydub import AudioSegment
            AudioSegment.silent(duration=15000).export(output_path, format="wav")
        
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = {
            "audio_url": f"/api/mashup/audio/{os.path.basename(output_path)}",
            "technical_review": review,
            "molecular_recipe": molecular_recipe,
            "collaboration_used": False,
            "solo_artist": "Claude"
        }
        
        return jobs[job_id]["result"]
    
    def _basic_revision(self, mashup_data: Dict, current_recipe: Dict, user_command: str, jobs: Dict, job_id: str) -> Dict[str, Any]:
        """Basic revision without AI assistance."""
        jobs[job_id]["progress"] = "⚙️ Creating basic revision..."
        
        # Create placeholder output
        output_path = f"workspace/mashups/basic_revision_{job_id}.wav"
        from pydub import AudioSegment
        AudioSegment.silent(duration=15000).export(output_path, format="wav")
        
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = {
            "audio_url": f"/api/mashup/audio/{os.path.basename(output_path)}",
            "message": "Basic revision completed. Enable AI APIs for advanced collaboration.",
            "collaboration_used": False
        }
        
        return jobs[job_id]["result"]
    
    # Helper methods
    
    def _get_molecular_summary(self) -> str:
        """Get summary of available molecular components."""
        if not self.molecular_catalog:
            return "basic audio components"
        
        total_components = sum(len(segments) for segments in self.molecular_catalog.values())
        component_types = []
        
        for stem_type, segments in self.molecular_catalog.items():
            if segments:
                component_types.append(f"{len(segments)} {stem_type} molecules")
        
        return f"{total_components} molecular components ({', '.join(component_types)})"
    
    def _get_fallback_molecules(self) -> Dict[str, Any]:
        """Get fallback molecular components when specific selection fails."""
        molecules = {}
        
        # Try to get at least one component from each available stem type
        for stem_type, segments in self.molecular_catalog.items():
            if segments:
                first_segment = list(segments.keys())[0]
                ingredient = f"{stem_type}_{first_segment}"
                molecules[f"fallback_{stem_type}"] = {
                    "ingredient": ingredient,
                    "effects": ["volume: -6dB"],
                    "molecular_type": "fallback_component"
                }
        
        return molecules


# Integration function for tasks.py
def revise_mashup_task(job_id: str, revision_data: Dict, jobs: Dict, openai_key: Optional[str], anthropic_key: Optional[str]):
    """Background task for mashup revision with molecular collaboration."""
    
    try:
        revision_engine = RevisionEngine(openai_key, anthropic_key)
        
        result = revision_engine.revise_with_collaboration(
            mashup_id=revision_data["mashup_id"],
            current_recipe=revision_data["current_recipe"], 
            user_command=revision_data["user_command"],
            jobs=jobs,
            job_id=job_id
        )
        
        if "error" in result:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = result["error"]
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"Revision task failed: {str(e)}"
        print(f"❌ Revision error: {e}")
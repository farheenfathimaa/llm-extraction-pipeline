"""
Sample document generator for testing the LLM Information Extraction Pipeline
"""

import os
import json
from pathlib import Path
from datetime import datetime

def create_sample_documents():
    """Create sample documents for testing"""
    
    # Create data directory
    data_dir = Path("data/sample_documents")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample research document 1: Climate Policy
    climate_policy_doc = """
    Climate Change Policy Analysis: A Comprehensive Review
    
    Executive Summary
    
    This comprehensive analysis examines the current state of climate change policies across major economies, 
    evaluating their effectiveness, implementation challenges, and potential for scaling. The research spans 
    multiple policy frameworks including carbon pricing mechanisms, renewable energy incentives, and 
    regulatory approaches to emissions reduction.
    
    Key Findings
    
    1. Carbon Pricing Effectiveness
    Carbon pricing mechanisms, including carbon taxes and cap-and-trade systems, have shown mixed results 
    across different jurisdictions. European Union's Emissions Trading System (EU ETS) has demonstrated 
    significant price volatility but has contributed to measurable emissions reductions in the power sector. 
    The analysis reveals that carbon prices need to reach $50-100 per ton CO2 to drive meaningful 
    industrial transformation.
    
    2. Renewable Energy Transition
    Feed-in tariffs and renewable energy standards have been instrumental in driving renewable energy 
    deployment. Countries with stable, long-term policy frameworks have achieved lower costs and faster 
    deployment rates. Germany's Energiewende program, despite initial challenges, has contributed to 
    dramatic cost reductions in solar and wind technologies globally.
    
    3. Regulatory Frameworks
    Command-and-control regulations, while less economically efficient than market-based mechanisms, 
    have proven effective in specific contexts. Vehicle emission standards have driven automotive 
    innovation, while building energy codes have improved construction practices. The analysis suggests 
    that hybrid approaches combining regulations with market incentives yield optimal results.
    
    4. International Cooperation
    The Paris Agreement has established a framework for global climate action, but national commitments 
    remain insufficient to meet temperature targets. Analysis of Nationally Determined Contributions 
    (NDCs) reveals an emissions gap that requires enhanced ambition and implementation support for 
    developing countries.
    
    Policy Recommendations
    
    1. Implement comprehensive carbon pricing with gradually increasing price floors
    2. Establish technology-neutral renewable energy standards
    3. Create green investment banks to address financing gaps
    4. Develop just transition frameworks for affected workers and communities
    5. Strengthen international climate finance mechanisms
    
    Implementation Challenges
    
    Political economy factors represent the primary barrier to effective climate policy implementation. 
    Vested interests in fossil fuel industries create resistance to policy changes, while the temporal 
    mismatch between policy costs and benefits complicates political decision-making. Public acceptance 
    varies significantly based on policy design and communication strategies.
    
    Economic modeling suggests that early action on climate policy generates net economic benefits 
    through avoided damages and innovation incentives. However, distributional impacts require careful 
    policy design to ensure equitable outcomes.
    
    Future Research Directions
    
    Further research is needed on policy interactions, behavioral responses to climate policies, and 
    the role of sub-national governments in climate action. Technology policy evaluation methodologies 
    require refinement to better capture spillover effects and learning curves.
    
    Conclusion
    
    Effective climate policy requires a portfolio approach combining market mechanisms, regulations, 
    and public investments. Success depends on policy design details, implementation capacity, and 
    political sustainability. The window for limiting warming to 1.5°C is narrowing, requiring 
    immediate scaling of proven policy approaches while continuing to innovate on policy design.
    """
    
    # Sample research document 2: Economic Analysis
    economic_analysis_doc = """
    Digital Economy Transformation: Impacts on Labor Markets and Policy Implications
    
    Introduction
    
    The digital transformation of the economy has accelerated significantly over the past decade, 
    fundamentally altering how businesses operate, workers perform their jobs, and value is created 
    and distributed. This analysis examines the multifaceted impacts of digitalization on labor 
    markets, with particular attention to policy implications for workforce development, social 
    protection, and economic governance.
    
    Digital Technology Adoption Patterns
    
    Enterprise adoption of digital technologies follows distinct patterns across sectors and firm sizes. 
    Large enterprises in technology-intensive sectors lead adoption, while small and medium enterprises 
    (SMEs) face significant barriers including limited technical expertise, capital constraints, and 
    uncertainty about returns on investment. The analysis reveals that successful digital transformation 
    requires complementary investments in organizational capabilities and human capital.
    
    Labor Market Impacts
    
    1. Job Displacement and Creation
    Automation and artificial intelligence are displacing routine tasks across occupations, affecting 
    both blue-collar and white-collar workers. However, new job categories are emerging in data analysis, 
    digital marketing, cybersecurity, and platform economy roles. The net employment effect varies by 
    region and time horizon, with short-term disruptions potentially offset by long-term job creation.
    
    2. Skill Requirements Evolution
    Digital transformation is changing skill requirements across occupations. Technical skills in 
    programming, data analysis, and digital literacy are increasingly valuable, while soft skills 
    like creativity, emotional intelligence, and complex problem-solving remain important. The analysis 
    suggests that successful workers will need to develop hybrid skill sets combining technical and 
    human capabilities.
    
    3. Work Organization Changes
    Digital platforms are enabling new forms of work organization, including remote work, gig economy 
    participation, and project-based collaboration. These changes offer flexibility benefits but also 
    create challenges for worker protection, benefits provision, and career development.
    
    Wage and Income Effects
    
    Digital transformation is contributing to wage polarization, with high-skilled workers experiencing 
    wage premiums while middle-skilled workers face downward pressure. Platform economy workers often 
    lack traditional employment benefits and face income volatility. The analysis finds that policy 
    interventions are needed to address these distributional consequences.
    
    Geographic Implications
    
    Digital technologies are reshaping economic geography, with some activities becoming more 
    location-independent while others concentrate in digital hubs. Rural areas face particular 
    challenges in accessing digital infrastructure and opportunities, potentially exacerbating 
    regional inequalities. Urban areas with strong digital ecosystems are attracting talent and 
    investment, creating agglomeration effects.
    
    Policy Recommendations
    
    1. Education and Training
    - Integrate digital literacy into curricula at all levels
    - Expand adult education and retraining programs
    - Support lifelong learning through portable training accounts
    - Strengthen partnerships between educational institutions and employers
    
    2. Social Protection Reform
    - Explore universal basic income pilot programs
    - Develop portable benefits systems for gig workers
    - Strengthen unemployment insurance for transitioning workers
    - Create social safety nets for platform economy participants
    
    3. Digital Infrastructure Investment
    - Ensure universal broadband access
    - Support digital payment systems and financial inclusion
    - Invest in cybersecurity infrastructure
    - Develop data governance frameworks
    
    4. Competition and Innovation Policy
    - Strengthen antitrust enforcement in digital markets
    - Support startup ecosystems and innovation hubs
    - Promote data portability and interoperability
    - Balance innovation incentives with consumer protection
    
    Implementation Challenges
    
    Policy implementation faces several challenges including rapid technological change, regulatory 
    lag, and international coordination requirements. The global nature of digital platforms 
    complicates national policy responses, while the pace of change strains traditional policy 
    development processes.
    
    Measurement and evaluation of digital economy policies require new methodologies and data 
    sources. Traditional economic indicators may not capture the full impacts of digital 
    transformation, necessitating development of new metrics and monitoring systems.
    
    International Dimensions
    
    Digital transformation has international implications through trade in digital services, 
    cross-border data flows, and global platform competition. Trade agreements increasingly 
    include digital provisions, while tax policy faces challenges from digital business models. 
    International cooperation is needed to address regulatory arbitrage and ensure fair competition.
    
    Conclusion
    
    The digital transformation of the economy presents both opportunities and challenges for 
    workers, businesses, and policymakers. Success in navigating this transformation requires 
    proactive policy responses that address market failures, support workforce adaptation, and 
    ensure broadly shared benefits. The policy agenda must balance innovation promotion with 
    worker protection, while addressing the distributional consequences of technological change.
    """
    
    # Sample research document 3: Public Health Policy
    public_health_doc = """
    Public Health Policy in the Post-Pandemic Era: Lessons Learned and Future Preparedness
    
    Abstract
    
    The COVID-19 pandemic has fundamentally transformed public health policy, revealing both strengths 
    and weaknesses in health systems worldwide. This comprehensive analysis examines policy responses 
    across multiple countries, evaluates their effectiveness, and identifies key lessons for future 
    pandemic preparedness and broader public health policy reform.
    
    Pandemic Response Evaluation
    
    1. Early Warning Systems
    Global health surveillance systems demonstrated significant gaps in early detection and reporting 
    of novel pathogens. Countries with robust surveillance infrastructure, including South Korea and 
    Taiwan, were able to respond more quickly and effectively. The analysis reveals that investment 
    in epidemiological capacity and international coordination mechanisms are critical for early 
    pandemic response.
    
    2. Public Health Measures
    Non-pharmaceutical interventions (NPIs) including lockdowns, mask mandates, and social distancing 
    measures showed varying effectiveness depending on timing, stringency, and public compliance. 
    Countries that implemented early, targeted measures with clear communication strategies achieved 
    better outcomes with lower economic costs. The analysis suggests that flexible, adaptive approaches 
    outperform rigid, one-size-fits-all policies.
    
    3. Healthcare System Resilience
    Health systems faced unprecedented stress, revealing capacity constraints and structural 
    vulnerabilities. Countries with stronger primary care systems and higher baseline healthcare 
    capacity managed surges more effectively. The analysis identifies critical infrastructure, 
    workforce, and supply chain elements that require strengthening.
    
    4. Vaccine Development and Distribution
    The rapid development of effective vaccines represented a remarkable scientific achievement, 
    but distribution revealed significant equity challenges. High-income countries secured vaccine 
    supplies while low-income countries faced delays and shortages. The analysis highlights the 
    need for improved global vaccine manufacturing and distribution systems.
    
    Health Equity Implications
    
    The pandemic disproportionately affected vulnerable populations, including racial and ethnic 
    minorities, low-income communities, and essential workers. Existing health disparities were 
    exacerbated by differential exposure risks, access to healthcare, and ability to implement 
    protective measures. Policy responses often failed to adequately address these equity concerns.
    
    Social determinants of health played a crucial role in pandemic outcomes, with housing conditions, 
    occupation, and neighborhood characteristics affecting transmission and mortality risks. The 
    analysis suggests that effective pandemic response requires addressing underlying social and 
    economic inequalities.
    
    Economic and Social Impacts
    
    Public health measures had significant economic and social consequences, including business 
    closures, unemployment, educational disruption, and mental health impacts. The analysis reveals 
    trade-offs between health protection and economic activity, with optimal policies balancing 
    multiple objectives through targeted interventions.
    
    Mental health impacts were substantial, particularly among children, adolescents, and isolated 
    populations. The pandemic highlighted gaps in mental health services and the need for integrated 
    approaches to physical and mental health policy.
    
    Future Preparedness Framework
    
    1. Surveillance and Early Warning
    - Strengthen global surveillance networks
    - Invest in laboratory capacity and genomic sequencing
    - Improve data sharing and international coordination
    - Develop rapid response protocols for novel pathogens
    
    2. Health System Strengthening
    - Build surge capacity in hospitals and public health departments
    - Develop flexible workforce deployment systems
    - Strengthen supply chains for medical countermeasures
    - Improve primary care and community health infrastructure
    
    3. Research and Development
    - Sustain investment in basic research and platform technologies
    - Develop rapid vaccine and therapeutic development capabilities
    - Support clinical trial infrastructure and regulatory capacity
    - Promote international research collaboration
    
    4. Risk Communication and Community Engagement
    - Improve public health communication strategies
    - Build trust through transparent, science-based messaging
    - Engage communities in preparedness planning
    - Address misinformation and disinformation
    
    Policy Integration and Coordination
    
    Effective pandemic response requires coordination across multiple sectors and levels of government. 
    The analysis reveals that countries with strong coordination mechanisms and clear authority structures 
    responded more effectively. Future preparedness requires developing institutional frameworks that 
    can rapidly mobilize resources and coordinate complex responses.
    
    International cooperation is essential for pandemic preparedness, given the global nature of 
    infectious disease threats. The analysis identifies needs for improved international health 
    regulations, financing mechanisms, and technology transfer systems.
    
    Implementation Priorities
    
    1. Immediate Actions (1-2 years)
    - Strengthen surveillance systems and laboratory capacity
    - Develop pandemic response plans with clear triggers and protocols
    - Build strategic reserves of medical countermeasures
    - Improve healthcare worker training and protection
    
    2. Medium-term Investments (3-5 years)
    - Expand manufacturing capacity for vaccines and therapeutics
    - Develop next-generation vaccine technologies
    - Strengthen primary care and community health systems
    - Address social determinants of health through intersectoral action
    
    3. Long-term Structural Changes (5-10 years)
    - Reform global health governance systems
    - Develop sustainable financing mechanisms for pandemic preparedness
    - Build resilient health systems that can address multiple threats
    - Create integrated One Health approaches linking human, animal, and environmental health
    
    Conclusion
    
    The COVID-19 pandemic has provided valuable lessons for public health policy, highlighting both 
    the importance of preparedness and the interconnected nature of health, economic, and social 
    systems. Future policy must address the root causes of health inequities while building resilient 
    systems capable of responding to emerging threats. Success requires sustained political commitment, 
    adequate resources, and global cooperation to ensure that all populations can benefit from 
    effective public health protection.
    """
    
    # Save documents
    documents = {
        "climate_policy_analysis.txt": climate_policy_doc,
        "digital_economy_transformation.txt": economic_analysis_doc,
        "public_health_pandemic_policy.txt": public_health_doc
    }
    
    for filename, content in documents.items():
        with open(data_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(documents)} sample documents in {data_dir}")
    return data_dir

def create_sample_config():
    """Create sample configuration file"""
    config = {
        "api_settings": {
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 4000,
            "temperature": 0.1,
            "timeout": 30
        },
        "processing_settings": {
            "chunk_size": 4000,
            "chunk_overlap": 200,
            "max_document_size": 10000000,  # 10MB
            "supported_formats": [".txt", ".pdf", ".docx", ".md"]
        },
        "chain_settings": {
            "max_retries": 3,
            "retry_delay": 1,
            "validation_threshold": 0.8
        },
        "output_settings": {
            "save_intermediate_results": True,
            "export_formats": ["json", "csv", "markdown"],
            "include_metadata": True
        }
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created sample configuration at {config_path}")
    return config_path

if __name__ == "__main__":
    # Create sample documents and configuration
    create_sample_documents()
    create_sample_config()
    print("Sample data setup complete!")
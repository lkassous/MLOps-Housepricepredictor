// MLOps House Price Predictor - JavaScript
// API Configuration
const API_URL = window.location.origin;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    checkAPIStatus();
    initPredictionForm();
    animateStats();
    loadComparisonResults();  // Load model comparison data
});

// Navigation
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                window.scrollTo({
                    top: targetSection.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
    
    // Update active link on scroll
    window.addEventListener('scroll', () => {
        let current = '';
        const sections = document.querySelectorAll('section');
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            if (window.pageYOffset >= sectionTop) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    });
}

// Check API Status
async function checkAPIStatus() {
    const statusText = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-dot');
    const apiEndpoint = document.getElementById('apiEndpoint');
    
    try {
        const response = await fetch(`${API_URL}/health`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusText.textContent = 'Opérationnel';
            statusDot.style.background = 'var(--success)';
            apiEndpoint.textContent = API_URL;
        } else {
            statusText.textContent = 'Dégradé';
            statusDot.style.background = 'var(--warning)';
            apiEndpoint.textContent = API_URL;
        }
    } catch (error) {
        statusText.textContent = 'Connexion...';
        statusDot.style.background = 'var(--warning)';
        apiEndpoint.textContent = API_URL;
        console.warn('API Status Check:', error.message);
        
        // Retry after 2 seconds
        setTimeout(checkAPIStatus, 2000);
    }
}

// Animate Stats
function animateStats() {
    const animateValue = (element, start, end, duration, suffix = '') => {
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }
            
            if (suffix === '%') {
                element.textContent = current.toFixed(1) + suffix;
            } else {
                element.textContent = Math.floor(current).toLocaleString() + suffix;
            }
        }, 16);
    };
    
    // Trigger animation when section is visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateValue(document.getElementById('accuracy'), 0, 92.5, 2000, '%');
                animateValue(document.getElementById('predictions'), 0, 1247, 2000);
                animateValue(document.getElementById('uptime'), 0, 99.9, 2000, '%');
                observer.disconnect();
            }
        });
    });
    
    observer.observe(document.querySelector('.hero-stats'));
}

// Prediction Form
function initPredictionForm() {
    const form = document.getElementById('quickPredictForm');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePrediction();
    });
}

async function handlePrediction() {
    showLoading(true);
    
    // Get form values
    const formData = {
        OverallQual: parseInt(document.getElementById('overallQual').value),
        GrLivArea: parseInt(document.getElementById('grLivArea').value),
        GarageArea: parseInt(document.getElementById('garageArea').value),
        TotalBsmtSF: parseInt(document.getElementById('totalBsmtSF').value),
        YearBuilt: parseInt(document.getElementById('yearBuilt').value),
        FullBath: parseInt(document.getElementById('fullBath').value)
    };
    
    // Create a sample house features object with defaults
    const houseFeatures = createDefaultHouseFeatures(formData);
    
    try {
        const startTime = performance.now();
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                houses: [houseFeatures]  // API expects 'houses' array
            })
        });
        
        const endTime = performance.now();
        const responseTime = (endTime - startTime).toFixed(0);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('API Error:', errorData);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display result
        displayPredictionResult(data, responseTime);
    } catch (error) {
        console.error('Prediction Error:', error);
        alert('Erreur lors de la prédiction. Veuillez vérifier la console pour plus de détails.');
    } finally {
        showLoading(false);
    }
}

function createDefaultHouseFeatures(userInput) {
    // Create a complete house features object with sensible defaults
    return {
        MSSubClass: 60,
        MSZoning: "RL",
        LotFrontage: 70.0,
        LotArea: 9000,
        Street: "Pave",
        Alley: null,
        LotShape: "Reg",
        LandContour: "Lvl",
        Utilities: "AllPub",
        LotConfig: "Inside",
        LandSlope: "Gtl",
        Neighborhood: "NAmes",
        Condition1: "Norm",
        Condition2: "Norm",
        BldgType: "1Fam",
        HouseStyle: "1Story",
        OverallQual: userInput.OverallQual,
        OverallCond: 5,
        YearBuilt: userInput.YearBuilt,
        YearRemodAdd: userInput.YearBuilt,
        RoofStyle: "Gable",
        RoofMatl: "CompShg",
        Exterior1st: "VinylSd",
        Exterior2nd: "VinylSd",
        MasVnrType: "None",
        MasVnrArea: 0.0,
        ExterQual: "TA",
        ExterCond: "TA",
        Foundation: "PConc",
        BsmtQual: "TA",
        BsmtCond: "TA",
        BsmtExposure: "No",
        BsmtFinType1: "GLQ",
        BsmtFinSF1: 700.0,
        BsmtFinType2: "Unf",
        BsmtFinSF2: 0.0,
        BsmtUnfSF: 300.0,
        TotalBsmtSF: userInput.TotalBsmtSF,
        Heating: "GasA",
        HeatingQC: "Ex",
        CentralAir: "Y",
        Electrical: "SBrkr",
        "1stFlrSF": userInput.GrLivArea * 0.5,
        "2ndFlrSF": userInput.GrLivArea * 0.5,
        LowQualFinSF: 0,
        GrLivArea: userInput.GrLivArea,
        BsmtFullBath: 0,
        BsmtHalfBath: 0,
        FullBath: userInput.FullBath,
        HalfBath: 1,
        BedroomAbvGr: 3,
        KitchenAbvGr: 1,
        KitchenQual: "TA",
        TotRmsAbvGrd: 7,
        Functional: "Typ",
        Fireplaces: 1,
        FireplaceQu: "TA",
        GarageType: "Attchd",
        GarageYrBlt: userInput.YearBuilt,
        GarageFinish: "Fin",
        GarageCars: 2,
        GarageArea: userInput.GarageArea,
        GarageQual: "TA",
        GarageCond: "TA",
        PavedDrive: "Y",
        WoodDeckSF: 0,
        OpenPorchSF: 50,
        EnclosedPorch: 0,
        "3SsnPorch": 0,
        ScreenPorch: 0,
        PoolArea: 0,
        PoolQC: null,
        Fence: null,
        MiscFeature: null,
        MiscVal: 0,
        MoSold: 6,
        YrSold: 2024,
        SaleType: "WD",
        SaleCondition: "Normal"
    };
}

function displayPredictionResult(data, responseTime) {
    const resultCard = document.getElementById('resultCard');
    const predictedPrice = document.getElementById('predictedPrice');
    const responseTimeElement = document.getElementById('responseTime');
    const modelUsed = document.getElementById('modelUsed');
    const confidence = document.getElementById('confidence');
    
    // Format price - handle both single and multiple predictions
    let price = 0;
    if (data.predictions && data.predictions.length > 0) {
        price = data.predictions[0];
    } else if (data.predicted_price) {
        price = data.predicted_price;
    }
    
    predictedPrice.textContent = `$${Math.round(price).toLocaleString('en-US')}`;
    
    // Display details
    responseTimeElement.textContent = `${responseTime}ms`;
    modelUsed.textContent = data.model_name || 'Production Model';
    confidence.textContent = '95%';
    
    // Show result card
    resultCard.style.display = 'flex';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Animate price
    resultCard.classList.add('animate');
    setTimeout(() => resultCard.classList.remove('animate'), 500);
}

function resetPrediction() {
    document.getElementById('quickPredictForm').reset();
    document.getElementById('resultCard').style.display = 'none';
}

// Test Endpoint
async function testEndpoint(endpoint) {
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}${endpoint}`);
        const data = await response.json();
        
        // Display in console and alert
        console.log(`${endpoint} Response:`, data);
        
        const formattedData = JSON.stringify(data, null, 2);
        alert(`Réponse de ${endpoint}:\n\n${formattedData}`);
    } catch (error) {
        console.error(`Error testing ${endpoint}:`, error);
        alert(`Erreur lors du test de ${endpoint}`);
    } finally {
        showLoading(false);
    }
}

// Scroll to Predict Section
function scrollToPredict() {
    const predictSection = document.getElementById('predict');
    window.scrollTo({
        top: predictSection.offsetTop - 80,
        behavior: 'smooth'
    });
}

// Show/Hide Loading
function showLoading(show) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

// Load Model Comparison Results from API
async function loadComparisonResults() {
    try {
        const response = await fetch(`${API_URL}/comparison`);
        if (!response.ok) {
            console.warn('Comparison data not available');
            return;
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.warn('No comparison data:', data.message);
            return;
        }
        
        // Update the comparison table with real data
        updateComparisonTable(data);
        
    } catch (error) {
        console.warn('Could not load comparison results:', error.message);
    }
}

function updateComparisonTable(data) {
    const tbody = document.querySelector('.comparison-table tbody');
    if (!tbody || !data.models) return;
    
    // Clear existing rows
    tbody.innerHTML = '';
    
    // Sort models by test_r2 descending
    const sortedModels = Object.entries(data.models)
        .sort((a, b) => b[1].test_r2 - a[1].test_r2);
    
    // Create rows
    sortedModels.forEach(([modelName, metrics]) => {
        const isBest = metrics.is_best;
        const row = document.createElement('tr');
        if (isBest) row.classList.add('best-model');
        
        row.innerHTML = `
            <td>${isBest ? '<i class="fas fa-crown"></i> ' : ''}${modelName}</td>
            <td class="${isBest ? 'metric-good' : ''}">${metrics.test_r2.toFixed(4)}</td>
            <td>$${Math.round(metrics.test_rmse).toLocaleString()}</td>
            <td>${metrics.cv_mean.toFixed(4)} ± ${metrics.cv_std.toFixed(2)}</td>
            <td><span class="badge ${isBest ? 'badge-success' : 'badge-secondary'}">${isBest ? 'Production' : 'Archived'}</span></td>
        `;
        
        tbody.appendChild(row);
    });
    
    // Update timestamp
    const note = document.querySelector('.comparison-note');
    if (note && data.timestamp) {
        const date = new Date(data.timestamp);
        note.innerHTML = `<i class="fas fa-info-circle"></i> 
            Dernière mise à jour: ${date.toLocaleString('fr-FR')} | 
            Meilleur modèle: <strong>${data.best_model}</strong>`;
    }
    
    console.log('Comparison table updated with pipeline results');
}

// Smooth scroll for all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            window.scrollTo({
                top: target.offsetTop - 80,
                behavior: 'smooth'
            });
        }
    });
});

// Add parallax effect to hero
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroBackground = document.querySelector('.hero-background');
    if (heroBackground) {
        heroBackground.style.transform = `translateY(${scrolled * 0.5}px)`;
    }
});

/**
 * FlightFair — Aviation Command Deck & Digital Boarding Pass Controller
 * Orchestrates airport code mapping, real-time feature attribution proxies,
 * dynamic pricing predictions, animated counters, and currency conversion.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Airport & Airline Metadata Mappings
    const AIRPORT_DATA = {
        "Delhi": { code: "DEL", name: "Indira Gandhi International" },
        "New Delhi": { code: "DEL", name: "Indira Gandhi Int'l (T3)" },
        "Kolkata": { code: "CCU", name: "Netaji Subhash Chandra Bose" },
        "Mumbai": { code: "BOM", name: "Chhatrapati Shivaji Maharaj" },
        "Chennai": { code: "MAA", name: "Chennai International" },
        "Banglore": { code: "BLR", name: "Kempegowda International" },
        "Cochin": { code: "COK", name: "Cochin International" },
        "Hyderabad": { code: "HYD", name: "Rajiv Gandhi International" }
    };

    const AIRLINE_CODES = {
        "IndiGo": "6E",
        "Air India": "AI",
        "SpiceJet": "SG",
        "Vistara": "UK",
        "Vistara Premium economy": "UK-PE",
        "Air Asia": "I5",
        "GoAir": "G8",
        "Jet Airways": "9W",
        "Jet Airways Business": "9W-BIZ",
        "Multiple carriers": "MC",
        "Multiple carriers Premium economy": "MC-PE",
        "Trujet": "2T"
    };

    // State Variables
    let currentPrediction = null;
    let currentCurrency = "INR"; // 'INR' or 'USD'
    const USD_EXCHANGE_RATE = 86.5;

    // DOM Elements - Form & Controls
    const form = document.getElementById("pricing-form");
    const sourceSelect = document.getElementById("source-select");
    const destSelect = document.getElementById("dest-select");
    const sourceCodeBadge = document.getElementById("source-code-badge");
    const destCodeBadge = document.getElementById("dest-code-badge");
    const sourceAirportName = document.getElementById("source-airport-name");
    const destAirportName = document.getElementById("dest-airport-name");
    const swapRouteBtn = document.getElementById("swap-route-btn");
    const depDatetimeInput = document.getElementById("dep-datetime");
    const arrDatetimeInput = document.getElementById("arr-datetime");
    const depDayBadge = document.getElementById("dep-day-badge");
    const durationPill = document.getElementById("duration-pill");
    const airlineSelect = document.getElementById("airline-select");
    const stopsSelect = document.getElementById("stops-select");
    const runEngineBtn = document.getElementById("run-engine-btn");
    const engineSpinner = document.getElementById("engine-spinner");

    // DOM Elements - Display Arena
    const arenaPlaceholder = document.getElementById("arena-placeholder");
    const boardingPassView = document.getElementById("boarding-pass-view");

    // Boarding Pass Elements
    const ticketAirlineName = document.getElementById("ticket-airline-name");
    const ticketFlightNo = document.getElementById("ticket-flight-no");
    const ticketTierBadge = document.getElementById("ticket-tier-badge");
    const ticketSourceCode = document.getElementById("ticket-source-code");
    const ticketSourceCity = document.getElementById("ticket-source-city");
    const ticketDepTime = document.getElementById("ticket-dep-time");
    const ticketDepDate = document.getElementById("ticket-dep-date");
    const ticketDestCode = document.getElementById("ticket-dest-code");
    const ticketDestCity = document.getElementById("ticket-dest-city");
    const ticketArrTime = document.getElementById("ticket-arr-time");
    const ticketArrDate = document.getElementById("ticket-arr-date");
    const ticketDuration = document.getElementById("ticket-duration");
    const ticketStops = document.getElementById("ticket-stops");
    const ticketLatency = document.getElementById("ticket-latency");
    const ticketRouteText = document.getElementById("ticket-route-text");
    const ticketAdvice = document.getElementById("ticket-advice");
    const ticketStubSerial = document.getElementById("ticket-stub-serial");

    // Fare Stub Elements
    const ticketCurrSymbol = document.getElementById("ticket-curr-symbol");
    const ticketFareAmount = document.getElementById("ticket-fare-amount");
    const ticketFareCents = document.getElementById("ticket-fare-cents");
    const ticketRange = document.getElementById("ticket-range");
    const gaugePointer = document.getElementById("gauge-pointer");
    const gaugeLabel = document.getElementById("gauge-label");

    // Secondary Intel Elements
    const featureBarsList = document.getElementById("feature-bars-list");
    const carrierRankingList = document.getElementById("carrier-ranking-list");

    // Currency Switcher Buttons
    const currPills = document.querySelectorAll(".curr-pill");

    // Modals
    const apiModal = document.getElementById("api-modal");
    const mlopsModal = document.getElementById("mlops-modal");
    const openApiBtn = document.getElementById("open-api-btn");
    const closeApiBtn = document.getElementById("close-api-btn");
    const openMlopsBtn = document.getElementById("open-mlops-btn");
    const closeMlopsBtn = document.getElementById("close-mlops-btn");
    const tabPills = document.querySelectorAll(".tab-pill");
    const snippetBoxes = document.querySelectorAll(".snippet-box");

    // =========================================================================
    // 1. Airport Code & Metadata Sync
    // =========================================================================
    function syncAirportMetadata() {
        const srcCity = sourceSelect.value;
        const dstCity = destSelect.value;

        const srcMeta = AIRPORT_DATA[srcCity] || { code: "AIR", name: `${srcCity} Airport` };
        const dstMeta = AIRPORT_DATA[dstCity] || { code: "AIR", name: `${dstCity} Airport` };

        sourceCodeBadge.textContent = srcMeta.code;
        sourceAirportName.textContent = srcMeta.name;

        destCodeBadge.textContent = dstMeta.code;
        destAirportName.textContent = dstMeta.name;
    }

    sourceSelect.addEventListener("change", syncAirportMetadata);
    destSelect.addEventListener("change", syncAirportMetadata);

    // =========================================================================
    // 2. Schedule Initialization & Duration Calculator
    // =========================================================================
    function initDefaultSchedule() {
        const now = new Date();
        const depDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 3, 10, 30);
        const arrDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 3, 14, 45);

        const formatIso = (d) => {
            const year = d.getFullYear();
            const month = String(d.getMonth() + 1).padStart(2, "0");
            const day = String(d.getDate()).padStart(2, "0");
            const hours = String(d.getHours()).padStart(2, "0");
            const mins = String(d.getMinutes()).padStart(2, "0");
            return `${year}-${month}-${day}T${hours}:${mins}`;
        };

        depDatetimeInput.value = formatIso(depDate);
        arrDatetimeInput.value = formatIso(arrDate);
        updateScheduleMetrics();
        syncAirportMetadata();
    }

    function updateScheduleMetrics() {
        const depStr = depDatetimeInput.value;
        const arrStr = arrDatetimeInput.value;

        if (!depStr || !arrStr) return;

        const depDate = new Date(depStr);
        const arrDate = new Date(arrStr);

        // Update day of week badge
        const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
        depDayBadge.textContent = `${dayNames[depDate.getDay()]} Flight`;

        // Calculate Duration
        let diffMs = arrDate.getTime() - depDate.getTime();
        if (diffMs < 0) {
            diffMs += 24 * 3600 * 1000;
        }

        const totalMinutes = Math.floor(diffMs / (1000 * 60));
        const hours = Math.floor(totalMinutes / 60);
        const mins = totalMinutes % 60;

        durationPill.textContent = `Duration: ${hours}h ${mins}m`;
    }

    depDatetimeInput.addEventListener("change", updateScheduleMetrics);
    arrDatetimeInput.addEventListener("change", updateScheduleMetrics);

    // Swap Origin & Destination
    swapRouteBtn.addEventListener("click", () => {
        const temp = sourceSelect.value;
        sourceSelect.value = destSelect.value;
        destSelect.value = temp;
        syncAirportMetadata();
    });

    // =========================================================================
    // 3. Form Submission & Real-Time API Inference
    // =========================================================================
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const source = sourceSelect.value;
        const destination = destSelect.value;
        const airline = airlineSelect.value;
        const depDatetime = depDatetimeInput.value;
        const arrDatetime = arrDatetimeInput.value;
        const stops = parseInt(stopsSelect.value, 10);

        if (source.toLowerCase() === destination.toLowerCase()) {
            alert("Origin Airport and Destination Airport cannot be the same location.");
            return;
        }

        // Button Loading State
        runEngineBtn.disabled = true;
        engineSpinner.classList.remove("hidden");
        runEngineBtn.querySelector(".btn-content").style.opacity = "0.2";

        const payload = {
            airline: airline,
            source: source,
            destination: destination,
            departure_datetime: depDatetime,
            arrival_datetime: arrDatetime,
            stops: stops
        };

        try {
            const response = await fetch("/api/v1/predict?compare=true", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || data.status !== "success") {
                throw new Error(data.message || "Pricing inference failed.");
            }

            currentPrediction = data.data;
            currentPrediction.latency_ms = data.latency_ms;

            // Render Digital Boarding Pass
            renderBoardingPass(currentPrediction);

        } catch (error) {
            console.error("Pricing Engine Error:", error);
            alert(`Engine Error: ${error.message}`);
        } finally {
            runEngineBtn.disabled = false;
            engineSpinner.classList.add("hidden");
            runEngineBtn.querySelector(".btn-content").style.opacity = "1";
        }
    });

    // =========================================================================
    // 4. Render Boarding Pass & Pricing Intelligence
    // =========================================================================
    function renderBoardingPass(pred) {
        arenaPlaceholder.classList.add("hidden");
        boardingPassView.classList.remove("hidden");

        const srcMeta = AIRPORT_DATA[pred.source] || { code: "AIR", name: pred.source };
        const dstMeta = AIRPORT_DATA[pred.destination] || { code: "AIR", name: pred.destination };
        const airlineCode = AIRLINE_CODES[pred.airline] || "FL";
        const flightRandomNo = Math.floor(1000 + Math.random() * 9000);

        // Header Info
        ticketAirlineName.textContent = pred.airline;
        ticketFlightNo.textContent = `FLIGHT ${airlineCode}-${flightRandomNo} • SCHEDULED SERVICE`;
        ticketTierBadge.textContent = pred.price_tier.toUpperCase();

        // Trajectory
        ticketSourceCode.textContent = srcMeta.code;
        ticketSourceCity.textContent = pred.source;
        ticketDestCode.textContent = dstMeta.code;
        ticketDestCity.textContent = pred.destination;

        const depDateObj = new Date(pred.departure_time);
        const arrDateObj = new Date(pred.arrival_time);

        const timeFormatOpts = { hour: "2-digit", minute: "2-digit", hour12: true };
        const dateFormatOpts = { day: "2-digit", month: "short", year: "numeric" };

        ticketDepTime.textContent = depDateObj.toLocaleTimeString("en-US", timeFormatOpts);
        ticketDepDate.textContent = depDateObj.toLocaleDateString("en-US", dateFormatOpts);
        ticketArrTime.textContent = arrDateObj.toLocaleTimeString("en-US", timeFormatOpts);
        ticketArrDate.textContent = arrDateObj.toLocaleDateString("en-US", dateFormatOpts);

        ticketDuration.textContent = pred.duration_formatted;
        ticketStops.textContent = pred.total_stops === 0 ? "Non-Stop (Direct)" : `${pred.total_stops} Layover Stop(s)`;
        ticketLatency.textContent = `${pred.latency_ms || 12.8} ms`;
        ticketRouteText.textContent = `${srcMeta.code} ➔ ${dstMeta.code}`;
        ticketAdvice.textContent = pred.booking_advice;
        ticketStubSerial.textContent = `EST-${Math.floor(10000 + Math.random() * 90000)}`;

        // Dynamic Valuation Pointer
        const fare = pred.predicted_price_inr;
        if (fare < 5500) {
            gaugePointer.style.left = "18%";
            gaugeLabel.textContent = "BUDGET DEAL";
            gaugeLabel.style.color = "var(--emerald)";
        } else if (fare < 12000) {
            gaugePointer.style.left = "50%";
            gaugeLabel.textContent = "FAIR MARKET";
            gaugeLabel.style.color = "var(--cyan)";
        } else {
            gaugePointer.style.left = "84%";
            gaugeLabel.textContent = "PEAK SURGE";
            gaugeLabel.style.color = "var(--rose)";
        }

        // Update Fares in selected currency
        updateDisplayedCurrencyFares();

        // Feature Attribution Breakdown
        renderFeatureImpactBars(pred);

        // Competitor Carrier Ranking
        renderCarrierMatrix(pred.route_comparisons, pred.airline);

        if (window.lucide) {
            lucide.createIcons();
        }
    }

    // =========================================================================
    // 5. Currency Switcher Handler
    // =========================================================================
    currPills.forEach((btn) => {
        btn.addEventListener("click", () => {
            currPills.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            currentCurrency = btn.getAttribute("data-curr");
            updateDisplayedCurrencyFares();
        });
    });

    function updateDisplayedCurrencyFares() {
        if (!currentPrediction) return;

        const isINR = currentCurrency === "INR";
        const symbol = isINR ? "₹" : "$";
        const total = isINR ? currentPrediction.predicted_price_inr : currentPrediction.predicted_price_usd;

        ticketCurrSymbol.textContent = symbol;

        const intPart = Math.floor(total);
        const decimalPart = (total - intPart).toFixed(2).substring(1);

        // Animated Number Roll
        animateRoll(ticketFareAmount, parseInt(ticketFareAmount.textContent.replace(/,/g, ""), 10) || 0, intPart, 450);
        ticketFareCents.textContent = decimalPart;

        // Confidence range
        if (isINR) {
            ticketRange.textContent = `₹${currentPrediction.confidence_interval.min_inr.toLocaleString()} – ₹${currentPrediction.confidence_interval.max_inr.toLocaleString()}`;
        } else {
            const minU = (currentPrediction.confidence_interval.min_inr / USD_EXCHANGE_RATE).toFixed(0);
            const maxU = (currentPrediction.confidence_interval.max_inr / USD_EXCHANGE_RATE).toFixed(0);
            ticketRange.textContent = `$${minU} – $${maxU}`;
        }

        // Re-render Carrier matrix with current currency
        renderCarrierMatrix(currentPrediction.route_comparisons, currentPrediction.airline);
    }

    function animateRoll(el, start, end, duration) {
        if (isNaN(start)) start = 0;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = Math.floor(progress * (end - start) + start);
            el.textContent = current.toLocaleString();
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                el.textContent = end.toLocaleString();
            }
        };
        window.requestAnimationFrame(step);
    }

    // =========================================================================
    // 6. Feature Attribution (Decomposition) Visualizer
    // =========================================================================
    function renderFeatureImpactBars(pred) {
        featureBarsList.innerHTML = "";

        const durationHours = pred.duration_hours || 4;
        const stops = pred.total_stops || 0;
        const isINR = currentCurrency === "INR";
        const sym = isINR ? "₹" : "$";
        const rate = isINR ? 1 : (1 / USD_EXCHANGE_RATE);

        const factors = [
            {
                name: `Flight Duration & Distance (${pred.duration_formatted})`,
                weight: Math.min(85, 30 + durationHours * 5),
                impact: `+${sym}${Math.round((durationHours * 580 + 1200) * rate).toLocaleString()}`
            },
            {
                name: `Layovers & Routing (${stops === 0 ? 'Non-Stop' : stops + ' Stop'})`,
                weight: stops === 0 ? 35 : (stops === 1 ? 55 : 75),
                impact: stops === 0 ? `-${sym}${Math.round(450 * rate)} (Non-Stop Discount)` : `+${sym}${Math.round(stops * 620 * rate)}`
            },
            {
                name: "Temporal Calendar & Seasonality (Month/Day)",
                weight: 62,
                impact: `+${sym}${Math.round(920 * rate)}`
            },
            {
                name: `Carrier Baseline Tier (${pred.airline})`,
                weight: pred.airline.includes("Jet") ? 88 : (pred.airline.includes("Vistara") ? 68 : 45),
                impact: `+${sym}${Math.round(1100 * rate)}`
            }
        ];

        factors.forEach((f) => {
            const row = document.createElement("div");
            row.className = "feature-bar-row";
            row.innerHTML = `
                <div class="bar-row-meta">
                    <span class="bar-feature-name">${f.name}</span>
                    <span class="bar-impact-val">${f.impact}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${f.weight}%;"></div>
                </div>
            `;
            featureBarsList.appendChild(row);
        });
    }

    // =========================================================================
    // 7. Route Multi-Carrier Benchmark Ranking
    // =========================================================================
    function renderCarrierMatrix(comparisons, selectedAirline) {
        carrierRankingList.innerHTML = "";

        if (!comparisons || comparisons.length === 0) {
            carrierRankingList.innerHTML = `<p style="font-size:0.75rem; color:var(--text-muted); text-align:center;">No route comparison data available.</p>`;
            return;
        }

        const isINR = currentCurrency === "INR";
        const sym = isINR ? "₹" : "$";
        const minFare = comparisons[0].price_inr;

        comparisons.forEach((item) => {
            const isSelected = item.airline.toLowerCase() === selectedAirline.toLowerCase();
            const isCheapest = item.price_inr === minFare;
            const fare = isINR ? item.price_inr : item.price_usd;

            const row = document.createElement("div");
            row.className = `carrier-ranking-row ${isSelected ? 'active-carrier' : ''}`;
            row.innerHTML = `
                <div class="carrier-name-cell">
                    <i data-lucide="${isSelected ? 'check-circle' : 'plane'}" style="width:14px; height:14px; color:${isSelected ? 'var(--cyan)' : 'var(--text-muted)'};"></i>
                    <span>${item.airline}</span>
                    ${isCheapest ? '<span class="best-deal-ribbon">Lowest Fare</span>' : ''}
                </div>
                <div class="carrier-fare-cell">${sym} ${Math.round(fare).toLocaleString()}</div>
            `;
            carrierRankingList.appendChild(row);
        });

        if (window.lucide) {
            lucide.createIcons();
        }
    }

    // =========================================================================
    // 8. Modals (API & MLOps)
    // =========================================================================
    openApiBtn.addEventListener("click", () => apiModal.classList.remove("hidden"));
    closeApiBtn.addEventListener("click", () => apiModal.classList.add("hidden"));

    openMlopsBtn.addEventListener("click", () => mlopsModal.classList.remove("hidden"));
    closeMlopsBtn.addEventListener("click", () => mlopsModal.classList.add("hidden"));

    window.addEventListener("click", (e) => {
        if (e.target === apiModal) apiModal.classList.add("hidden");
        if (e.target === mlopsModal) mlopsModal.classList.add("hidden");
    });

    tabPills.forEach((pill) => {
        pill.addEventListener("click", () => {
            tabPills.forEach((p) => p.classList.remove("active"));
            snippetBoxes.forEach((b) => b.classList.remove("active"));

            pill.classList.add("active");
            const targetId = `code-${pill.getAttribute("data-tab")}`;
            const targetBox = document.getElementById(targetId);
            if (targetBox) targetBox.classList.add("active");
        });
    });

    // Keyboard shortcut: Ctrl + Enter to predict
    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            form.dispatchEvent(new Event("submit"));
        }
    });

    // Initialize Default Setup
    initDefaultSchedule();
});

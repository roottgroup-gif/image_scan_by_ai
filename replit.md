# Overview

This is an AI Image Detection Checker application that allows users to upload images and determine whether they are AI-generated or real. The application features a React frontend with a Node.js/Express backend, using a PostgreSQL database for data persistence. Users can upload images through a modern web interface, receive real-time analysis results with confidence scores, and view detailed detection indicators that influenced the classification decision.

# User Preferences

Preferred communication style: Simple, everyday language.
Design preference: Shadow-free, smooth design throughout the application (no shadow effects like shadow-sm, shadow-lg, shadow-xl).

# System Architecture

## Frontend Architecture
- **Framework**: React with TypeScript using Vite as the build tool
- **UI Components**: Radix UI primitives with shadcn/ui component library for consistent, accessible design
- **Styling**: Tailwind CSS with CSS variables for theming support (light/dark mode)
- **State Management**: TanStack Query (React Query) for server state management and caching
- **Routing**: Wouter for lightweight client-side routing
- **Form Handling**: React Hook Form with Zod validation for type-safe form management

## Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **API Design**: RESTful API with structured error handling and request logging
- **File Processing**: Multer for multipart file uploads with Sharp for image processing
- **Session Management**: Express sessions with PostgreSQL session store

## Data Storage
- **Database**: PostgreSQL with Drizzle ORM for type-safe database operations
- **Schema Management**: Drizzle Kit for migrations and schema management
- **Connection**: Neon serverless PostgreSQL adapter for cloud database connectivity
- **Storage Strategy**: In-memory fallback storage for development with interface-based architecture for easy switching

## Authentication & Sessions
- **Session Storage**: PostgreSQL-backed sessions using connect-pg-simple
- **User Management**: Basic user registration and authentication system
- **Security**: Password hashing and secure session management

## AI Detection System
- **Enhanced Neural Networks**: Transfer learning with ResNet50 for deep learning-based AI detection
- **Hybrid Analysis**: Combines CNN deep learning with traditional computer vision forensics
- **Multi-Algorithm Detection**: 6 advanced algorithms including texture analysis, frequency domain, compression forensics
- **Processing Pipeline**: Multi-stage analysis including metadata extraction, noise pattern detection, and artifact identification
- **Result Structure**: Confidence scoring, classification labels, and detailed indicator reporting
- **Performance Tracking**: Processing time measurement and optimization

## Monetization System
- **Pre-Analysis Ad Modal**: Interactive modal displaying premium features and upgrade options before analysis
- **Timed Ad Experience**: 8-second countdown with premium service promotion
- **Premium Upselling**: Showcases batch processing, API access, and priority support features
- **Strategic Ad Placement**: Ads shown at optimal engagement moments (before analysis starts)

## Development Workflow
- **Build System**: Vite for frontend bundling with hot module replacement
- **Development Server**: Concurrent frontend and backend development with proxy setup
- **Production Build**: ESBuild for backend compilation and Vite for frontend optimization
- **Type Safety**: Shared TypeScript types between frontend and backend via shared schema

# External Dependencies

## Database Services
- **Neon Database**: Serverless PostgreSQL hosting with connection pooling
- **Drizzle**: Type-safe ORM with PostgreSQL dialect support

## Image Processing
- **Sharp**: High-performance image processing for metadata extraction and format conversion
- **Multer**: Express middleware for handling multipart/form-data file uploads

## UI Framework
- **Radix UI**: Accessible, unstyled UI primitives for complex components
- **Lucide React**: Consistent icon library with React components
- **Tailwind CSS**: Utility-first CSS framework with custom design system

## Development Tools
- **Replit Integration**: Development environment with runtime error handling and debugging tools
- **Vite Plugins**: Development enhancement with error overlays and hot reloading
- **TypeScript**: Static type checking across the entire application stack

## State Management
- **TanStack Query**: Server state synchronization with caching and background updates
- **Wouter**: Minimalist client-side routing solution

## Session & Security
- **connect-pg-simple**: PostgreSQL session store for Express sessions
- **Express Session**: Server-side session management with secure cookie handling

# Recent Changes

## Complete Node.js to Python Flask Migration (January 2025)
- **Complete Backend Migration**: Successfully converted from Node.js/Express to pure Python Flask
- **Local Storage Only**: Removed all database dependencies, using in-memory local storage
- **Design Preservation**: Maintained all original UI elements, styling, and user experience
- **AI Detection System**: Integrated computer vision-based AI detection algorithms
- **Pure Python Stack**: Eliminated all Node.js dependencies and Vite build system
- **Simplified Architecture**: Single Python Flask application with HTML templates and static assets
- **Workflow Migration**: Updated from npm/Node.js workflow to Python Flask server on port 5000

## Shadow Removal Implementation (January 2025)
- **Complete Shadow Elimination**: Removed all shadow effects across the entire application for a clean, smooth design
- **Core Components Updated**: Modified card.tsx, dialog.tsx, toast.tsx, command.tsx UI components
- **Page-Level Updates**: Updated all pages (home, about, how-it-works, faq) to remove shadow classes
- **CSS Updates**: Cleaned index.css and component styles to eliminate all shadow variants
- **Design Consistency**: Maintained gradient backgrounds and backdrop blur effects while removing all shadow styling

## Professional SEO Implementation (January 2025)
- **Comprehensive Meta Tags**: Added title, description, keywords, robots, author, and language meta tags
- **Open Graph Tags**: Full Facebook/social media optimization with titles, descriptions, and images
- **Twitter Cards**: Large image cards with proper meta tags for Twitter sharing
- **Structured Data**: JSON-LD schema markup for WebApplication, Organization, HowTo, FAQ, and Breadcrumb schemas
- **Technical SEO**: Canonical URLs, theme colors, sitemap.xml, robots.txt, and PWA manifest
- **Page-Specific SEO**: Dynamic meta tag updates per page with unique titles, descriptions, and structured data
- **Performance SEO**: DNS prefetch, preconnect, and mobile optimization meta tags
- **Search Features**: Website schema with search action for enhanced search engine integration

## AI Generated Danger Color Implementation (January 2025)
- **Warning Classification**: AI Generated results now display in red/danger colors to indicate potential risk
- **Badge Styling**: AI Generated badge uses red gradient (from-red-500 to-red-600) with red border
- **Visual Indicators**: Wave animations, image borders, and status badges use red colors for AI Generated content
- **Progress Bars**: Confidence score progress bars display in red gradients for AI Generated classifications
- **Alert Icons**: AlertTriangle icons replace CheckCircle for AI Generated results to emphasize warning nature
- **Consistent Warning Theme**: All UI elements consistently use red/danger colors when AI content is detected

## Professional PDF Report Implementation (January 2025)
- **Visual Report Design**: HTML-based reports with analyzed image, brand colors, and professional styling
- **Brand Consistency**: Uses teal/mint green gradients (#5bc0be, #4a9a98) and red danger colors for AI Generated content
- **Comprehensive Layout**: Image preview, classification badge, confidence visualization, and technical analysis grid
- **Print-to-PDF Functionality**: Browser print dialog for professional PDF generation with proper formatting
- **Visual Indicators**: Color-coded classification badges, progress bars, and warning sections
- **Complete Analysis**: Includes methodology, key indicators, disclaimers, and technical specifications
- **Professional Styling**: Corporate design with gradients, structured sections, and responsive layout

## Vertical Google Ads Implementation (January 2025)
- **Side Ad Placement**: Fixed vertical ad spaces on left and right sides of main content
- **Responsive Design**: Ads only visible on XL screens (1280px+) to maintain mobile experience
- **Professional Styling**: Glassmorphism design with brand colors and backdrop blur effects
- **Standard Format**: 160x600 Skyscraper ad format for optimal Google AdSense integration
- **Non-Intrusive**: Fixed positioning ensures ads don't interfere with main content flow
- **Brand Consistency**: Uses teal gradient colors matching overall design theme